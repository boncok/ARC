# -*- coding: utf-8 -*-
"""Thread-based prefetch cho Ultralytics InfiniteDataLoader (workers=0 trong sandbox).

Sandbox chan multiprocessing.Pipe nen DataLoader num_workers>0 khong chay duoc.
Thay vao do: 1 pool thread goi dataset.__getitem__ song song, 1 producer thread
ghep batch va day vao queue; main thread vua lay batch vua chay GPU -> overlap
augment (CPU) voi forward/backward (XPU).
"""
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data._utils.collate import default_collate


class ThreadedInfiniteIterator:
    def __init__(self, loader, workers=6, prefetch=4):
        self.loader = loader
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn or default_collate
        self.bs_iter = iter(loader.batch_sampler)
        self.ex = ThreadPoolExecutor(max_workers=workers)
        self.q = queue.Queue(maxsize=prefetch)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            while not self.stop.is_set():
                idxs = next(self.bs_iter)
                futs = [self.ex.submit(self.dataset.__getitem__, i) for i in idxs]
                items = [f.result() for f in futs]
                if self.stop.is_set():
                    break
                self.q.put(self.collate_fn(items))
        except Exception as e:  # propagate to consumer
            try:
                self.q.put(e)
            except Exception:
                pass
        finally:
            try:
                self.q.put(StopIteration)
            except Exception:
                pass

    def __next__(self):
        x = self.q.get()
        if x is StopIteration:
            raise StopIteration
        if isinstance(x, Exception):
            raise x
        return x

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.ex.shutdown(wait=False, cancel_futures=True)


def patch_infinite_loader(workers=6, prefetch=4):
    from ultralytics.data.build import InfiniteDataLoader

    if getattr(InfiniteDataLoader, "_thread_patched", False):
        return

    orig_iter = InfiniteDataLoader.__iter__
    orig_reset = InfiniteDataLoader.reset

    def __iter__(self):
        if not getattr(self.dataset, "augment", False):
            yield from orig_iter(self)
            return
        it = getattr(self, "_thread_iter", None)
        if it is None:
            it = ThreadedInfiniteIterator(self, workers=workers, prefetch=prefetch)
            object.__setattr__(self, "_thread_iter", it)
        for _ in range(len(self)):
            yield next(it)

    def reset(self):
        it = getattr(self, "_thread_iter", None)
        if it is not None:
            it.close()
            object.__setattr__(self, "_thread_iter", None)
        orig_reset(self)

    InfiniteDataLoader.__iter__ = __iter__
    InfiniteDataLoader.reset = reset
    InfiniteDataLoader._thread_patched = True
    print(f"[thread_loader] patched InfiniteDataLoader workers={workers} prefetch={prefetch}", flush=True)
