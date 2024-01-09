import pkg_resources

installed_packages = [pkg.key for pkg in pkg_resources.working_set]
installed_packages.sort()

for package in installed_packages:
    print(package)
