/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <callbacks.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
class Transport {
  int fd = -1;
  std::mutex mu;
  struct Region {
    uint8_t *host;
    uint64_t size;
  };
  std::map<uint64_t, Region> regions;
  uint64_t next = 0x81000000, ring = 0, head = 0, complete = 0;
  static constexpr uint64_t ctrl = 0x40000000, cp = 0x40101000;
  std::string request(const std::string &cmd) {
    std::string out = cmd + "\n";
    size_t done = 0;
    while (done < out.size()) {
      auto n = write(fd, out.data() + done, out.size() - done);
      if (n <= 0)
        throw std::runtime_error("qtest write failed");
      done += n;
    }
    std::string line;
    char c;
    while (read(fd, &c, 1) == 1) {
      if (c == '\n')
        break;
      line += c;
    }
    if (line.compare(0, 2, "OK"))
      throw std::runtime_error(line);
    return line.size() > 3 ? line.substr(3) : "";
  }
  uint32_t read32(uint64_t a) {
    char b[80];
    std::snprintf(b, sizeof(b), "readl 0x%llx", (unsigned long long)a);
    return std::stoul(request(b), nullptr, 0);
  }
  void write32(uint64_t a, uint32_t v) {
    char b[80];
    std::snprintf(b, sizeof(b), "writel 0x%llx 0x%x", (unsigned long long)a, v);
    request(b);
  }
  void sync(uint64_t addr, bool pull) {
    auto it = regions.find(addr);
    if (it == regions.end())
      throw std::runtime_error("unknown host region");
    auto &r = it->second;
    for (uint64_t off = 0; off < r.size; off += 4096) {
      uint64_t n = std::min<uint64_t>(4096, r.size - off);
      char b[100];
      std::snprintf(b, sizeof(b), "%s 0x%llx 0x%llx", pull ? "read" : "write",
                    (unsigned long long)(addr + off), (unsigned long long)n);
      if (pull) {
        auto s = request(b);
        if (s.size() != 2 + n * 2)
          throw std::runtime_error("short qtest memory read");
        for (uint64_t j = 0; j < n; ++j) {
          char byte[3] = {s[2 + j * 2], s[3 + j * 2], 0};
          r.host[off + j] = std::strtoul(byte, nullptr, 16);
        }
      } else {
        std::string s = std::string(b) + " 0x";
        static const char h[] = "0123456789abcdef";
        for (uint64_t j = 0; j < n; ++j) {
          auto v = r.host[off + j];
          s += h[v >> 4];
          s += h[v & 15];
        }
        request(s);
      }
    }
  }

public:
  Transport() {
    const char *path = std::getenv("GPGPU_QTEST_SOCKET");
    if (!path)
      throw std::runtime_error("GPGPU_QTEST_SOCKET required");
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (std::strlen(path) >= sizeof(a.sun_path))
      throw std::runtime_error("socket path too long");
    std::strcpy(a.sun_path, path);
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0 || connect(fd, (sockaddr *)&a, sizeof(a)))
      throw std::runtime_error("qtest connect failed");
    uint64_t cfg = 0;
    for (unsigned slot = 0; slot < 32; ++slot) {
      uint64_t a = 0x30000000ull + (slot << 15);
      if (read32(a) == 0x13371234) {
        cfg = a;
        break;
      }
    }
    if (!cfg)
      throw std::runtime_error("GPGPU PCI device not found");
    write32(cfg + 0x10, ctrl);
    write32(cfg + 0x14, 0);
    write32(cfg + 0x18, 0x44000000);
    write32(cfg + 0x1c, 0);
    write32(cfg + 0x20, 0x40100000);
    write32(cfg + 4, 6);
    write32(ctrl + 0x100, 1);
  }
  ~Transport() {
    for (auto &p : regions)
      std::free(p.second.host);
    if (fd >= 0)
      close(fd);
  }
  int reg_write(uint32_t off, uint32_t v) {
    std::lock_guard<std::mutex> lock(mu);
    try {
      auto half = [off, v](uint64_t &a, uint32_t lo) {
        if (off == lo)
          a = (a & 0xffffffff00000000ull) | v;
        if (off == lo + 4)
          a = (a & 0xffffffffull) | (uint64_t(v) << 32);
      };
      half(ring, 0x100);
      half(head, 0x108);
      half(complete, 0x110);
      /* Queue doorbells publish CPU writes to the command ring. */
      if (off == 0x124 && ring)
        sync(ring, false);
      write32(cp + off, v);
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "qtest: %s\n", e.what());
      return -1;
    }
  }
  int reg_read(uint32_t off, uint32_t *v) {
    std::lock_guard<std::mutex> lock(mu);
    try {
      *v = read32(cp + off);
      if (head)
        sync(head, true);
      if (complete)
        sync(complete, true);
      if (read32(ctrl + 0x108))
        throw std::runtime_error("GPU execution fault");
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "qtest: %s\n", e.what());
      return -1;
    }
  }
  int alloc(uint64_t n, void **p, uint64_t *a) {
    std::lock_guard<std::mutex> lock(mu);
    n = (n + 63) & ~63ull;
    if (!n || n > 0x08000000 || next + n > 0x89000000)
      return -1;
    *p = std::calloc(1, n);
    if (!*p)
      return -1;
    *a = next;
    next += n;
    regions[*a] = {(uint8_t *)*p, n};
    try {
      sync(*a, false);
      return 0;
    } catch (...) {
      return -1;
    }
  }
  int release(uint64_t a) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = regions.find(a);
    if (it == regions.end())
      return -1;
    if (ring == a)
      ring = 0;
    if (head == a)
      head = 0;
    if (complete == a)
      complete = 0;
    std::free(it->second.host);
    regions.erase(it);
    return 0;
  }
  int refresh(uint64_t a, bool pull) {
    std::lock_guard<std::mutex> lock(mu);
    try {
      sync(a, pull);
      return 0;
    } catch (...) {
      return -1;
    }
  }
};
} // namespace
extern "C" int vx_dev_init(callbacks_t *c) {
  if (!c)
    return -1;
  c->dev_open = [](void **p) {
    try {
      *p = new Transport;
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "qtest: %s\n", e.what());
      return -1;
    }
  };
  c->dev_close = [](void *p) {
    delete (Transport *)p;
    return 0;
  };
  c->cp_reg_write = [](void *p, uint32_t o, uint32_t v) {
    return ((Transport *)p)->reg_write(o, v);
  };
  c->cp_reg_read = [](void *p, uint32_t o, uint32_t *v) {
    return ((Transport *)p)->reg_read(o, v);
  };
  c->host_mem_alloc = [](void *p, uint64_t n, void **h, uint64_t *a) {
    return ((Transport *)p)->alloc(n, h, a);
  };
  c->host_mem_free = [](void *p, uint64_t a) {
    return ((Transport *)p)->release(a);
  };
  c->host_mem_push = [](void *p, uint64_t a) {
    return ((Transport *)p)->refresh(a, false);
  };
  c->host_mem_pull = [](void *p, uint64_t a) {
    return ((Transport *)p)->refresh(a, true);
  };
  return 0;
}
