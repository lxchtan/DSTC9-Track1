import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from baseline_modify import generate as main
    else:
        from baseline_modify import main

    main.main()
