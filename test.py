from tqdm import tqdm
import time
pbar = tqdm(range(1000))
for char in pbar:
    time.sleep(0.5)
    # print(char)
    pbar.set_description("Processing %s" % char)