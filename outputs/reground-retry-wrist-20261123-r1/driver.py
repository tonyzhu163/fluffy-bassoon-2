"""Per-device whole-pair batch publication. Keep native spawned simulator imports light."""
from pathlib import Path
import sys


def main():
    import reference as ref
    ref.configure()
    if sys.argv[1] == "cpu":
        if sys.argv[2] == "readback":
            import summarize
            summarize.selftest(Path(sys.argv[3]))
        elif sys.argv[2] == "audit":
            import audit
            audit.run(Path(sys.argv[3]))
        elif sys.argv[2] == "layouts":
            import layouts
            layouts.run(Path(sys.argv[3]))
        else:
            import qualify
            qualify.run(Path(sys.argv[3]), sys.argv[2])
    elif sys.argv[1] == "lane":
        import worker
        worker.run(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "summarize":
        import summarize
        summarize.report(Path(sys.argv[2]),sys.argv[3])
    else:
        raise ValueError(sys.argv[1])


if __name__ == "__main__":
    main()
