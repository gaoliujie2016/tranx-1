from termcolor import colored


def info(*msg):
    s = "[I] " + " ".join(list(map(str, msg)))
    print(colored(s, "green"))


def debug(*msg):
    s = "[D] " + " ".join(list(map(str, msg)))
    print(colored(s, "magenta"))


def exception(*msg):
    s = "[X] " + " ".join(list(map(str, msg)))
    print(colored(s, "yellow"))


def error(*msg, do_raise=False):
    s = "[E] " + " ".join(list(map(str, msg)))
    if do_raise:
        raise RuntimeError(s)
    else:
        print(colored(s, "red"))
