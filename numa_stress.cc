#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <numa.h>
#include <regex>
#include <thread>
#include <vector>
#include <numaif.h>

constexpr std::size_t ARRAY_SIZE = 1ULL << 30; // 1 GiB per remote node
constexpr int HOME_NODE = 1;                   // threads run on NUMA-1
constexpr int TOTAL_NODES = 16;                // 0-15 on this platform

/* ───────────────── helpers ───────────────── */

static std::size_t cache_line() {
  std::ifstream f(
      "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
  std::size_t v = 64;
  if (f) {
    // read cache line size from system config
    // TODO: maybe find some API that can retrieve this portably in case you
    // want to test this on BSD
    f >> v;
  }
  return v ? v : 64;
}

// support for durations like 1s 1H etc
static std::size_t parse_secs(const char *arg) {
  std::regex re(R"(^(\d+)([smh])$)");
  std::cmatch m;
  if (!std::regex_match(arg, m, re)) {
    std::cerr << "duration format: <number>[s|m|h]\n";
    std::exit(1);
  }
  std::size_t n = std::stoul(m[1]);
  char u = m[2].str()[0];

  return u == 'h' ? n * 3600 : (u == 'm' ? n * 60 : n);
}

static void pin_to_node(int node) {
  if (numa_run_on_node(node) != 0) {
    std::perror("numa_run_on_node");
    std::exit(1);
  }
}

/* ───────────────── worker ───────────────── */

struct stats {
  double gbps = 0;
};

std::mutex out_mtx;

static void worker(int remote_node, std::size_t secs, const std::size_t line,
                   stats &st) {
  // TODO: some nice scope-guard style error handling would be better
  numa_set_strict(1);

  void *raw_buf = nullptr;
  // allocate on cache line boundary
  // so that we favor full line writes
  if (posix_memalign(&raw_buf, line, ARRAY_SIZE) != 0) {
    std::cerr << "aligned allocation failed\n";
    std::exit(1);
  }

  // since the allocation with memalign never faulted a page into existence
  // the following policy-based assignment will work, otherwise, if a write
  // happens before this to any of the pages referred to by the prev posix
  // memalign => kaboom
  numa_tonode_memory(raw_buf, ARRAY_SIZE, remote_node);

  // ASSERT correct alignment of start of buffer memory
  // TODO: maybe wrap this in a nice wrapper that can be disabled
  // on demand through a build flag
  char *temp_buf = static_cast<char *>(raw_buf);
  uintptr_t addr = reinterpret_cast<uintptr_t>(temp_buf);

  // Calculate alignment offset needed
  // and correctly set buf as start of writes
  uintptr_t misalignment = addr % line;
  char *buf = misalignment ? temp_buf + (line - misalignment) : temp_buf;

  // Calculate effective array size after alignment adjustment
  std::size_t alignment_loss = buf - temp_buf;
  std::size_t effective_array_size = ARRAY_SIZE - alignment_loss;

  // Round down to nearest cache line boundary
  effective_array_size = (effective_array_size / line) * line;

  if (effective_array_size < line) {
    std::cerr << "Effective array size too small after alignment correction\n";
    free(raw_buf);
    std::exit(1);
  }

  /* first-touch to force pages onto remote_node */
  std::fill(buf, buf + effective_array_size, 0);

  // Verify migration succeeded for a sample page
  void *sample_page = raw_buf;
  int status;
  if (move_pages(0, 1, &sample_page, NULL, &status, MPOL_MF_MOVE) != 0) {
    // handle errno reporting via perror, not cerr but whatever
    perror("move_pages verification failed");
    free(raw_buf);
    std::exit(1);
  }
  if (status != remote_node) {
    std::cerr << "Page not on target node " << remote_node << " (got " << status
              << ")\n";
    free(raw_buf);
    std::exit(1);
  }

  /* streaming pattern */
  std::size_t offset = 0;
  std::uint8_t val = 0;
  std::size_t bytes = 0;

  const auto t0 = std::chrono::steady_clock::now();
  auto t_prev = t0;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - t0).count() >=
        secs)
      break;

#if USE_ASM_AVX512
    __m512i pattern = _mm512_set1_epi8(static_cast<char>(val));
    asm volatile("vmovdqa64 %[val], (%[addr])\n\t" // Single 64-byte write
                 :                                 // No outputs
                 : [addr] "r"(array + offset), [val] "v"(pattern) // "v" for ZMM
                 : "memory");

    /* advance */
#elif USE_ASM_SSE
    asm volatile(
        "movdqa %[val], (%[addr])\n\t" // Write 16 bytes (XMM register)
        "movdqa %[val], 16(%[addr])\n\t"
        "movdqa %[val], 32(%[addr])\n\t"
        "movdqa %[val], 48(%[addr])\n\t" // Completes 64 bytes
        // Optional: clflush (%[addr]) for flush or movntdq for non-temporal
        :                                               // No outputs
        : [addr] "r"(array + line), [val] "x"(value128) // "x" for XMM
        : "memory"                                      // Memory clobber
    );
#else
    /* prepare 64-byte pattern */
    __m512i pattern = _mm512_set1_epi8(static_cast<char>(val));
    /* 64-B non-temporal store  (vmovntdq)   */
    _mm512_stream_si512(reinterpret_cast<__m512i *>(buf + offset), pattern);

#endif

    /* advance by 1 line since we're incrementing by 64B every time*/
    offset += line;

    // loop if at the end of the array
    if (offset >= effective_array_size) {
      offset = 0;
      ++val;
    }

    bytes += line;

    /* 1-second progress report */
    if (std::chrono::duration_cast<std::chrono::seconds>(now - t_prev)
            .count() >= 1) {
      double elapsed = std::chrono::duration<double>(now - t0).count();
      double gbs = double(bytes) / (1ULL << 30) / elapsed;
      {
        std::lock_guard<std::mutex> lk(out_mtx);
        std::cout << "node " << remote_node << "  " << int(elapsed) << " s  "
                  << gbs << " GB/s\n";
      }
      t_prev = now;
    }
  }

  st.gbps = double(bytes) / (1ULL << 30) / secs;
  numa_free(buf, ARRAY_SIZE);
}

/* ───────────────── main ───────────────── */

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <duration>\n";
    return 1;
  }

  if (numa_available() < 0) {
    std::cerr << "libnuma: NUMA not available\n";
    return 1;
  }
  const std::size_t run_secs = parse_secs(argv[1]);
  const std::size_t line = cache_line(); // 64 B on SKX
  if (ARRAY_SIZE % line) {
    std::cerr << "ARRAY_SIZE must be multiple of cache line\n";
    return 1;
  }

  pin_to_node(HOME_NODE); // pin main + workers on NUMA-1

  /* build list of remote nodes */
  std::vector<int> rem;
  for (int n = 0; n < TOTAL_NODES; ++n)
    if (n != HOME_NODE)
      rem.push_back(n);

  std::vector<stats> all(rem.size());
  std::vector<std::jthread> th;

  for (std::size_t i = 0; i < rem.size(); ++i)
    th.emplace_back(worker, rem[i], run_secs, line, std::ref(all[i]));

  th.clear(); // joins

  double agg = 0;
  for (auto &s : all)
    agg += s.gbps;
  std::cout << "\naggregate write bandwidth: " << agg << " GB/s\n";
}
