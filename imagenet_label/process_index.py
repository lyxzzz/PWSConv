fname = "train_100_class_labeled.txt"

with open(fname, "r") as f:
    lines = f.readlines()

with open(fname, "w") as f:
    prev = int(lines[0].split(" ")[1])
    cnt = 0
    for l in lines:
        sp = l.split(" ")
        clsnum = int(sp[1])
        if clsnum != prev:
            cnt += 1
            prev = clsnum
        f.write("{} {}\n".format(sp[0], cnt))
