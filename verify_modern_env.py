import importlib.util

pkgs = ['torch', 'ultralytics', 'cv2', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
for pkg in pkgs:
    spec = importlib.util.find_spec(pkg)
    print(pkg, 'FOUND' if spec else 'MISSING')
    if spec:
        mod = __import__(pkg)
        print('  version', getattr(mod, '__version__', 'unknown'))
