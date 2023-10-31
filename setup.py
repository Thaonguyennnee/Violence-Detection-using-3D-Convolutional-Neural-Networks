import pkg_resources

def main():
    # Get a list of all installed distributions
    installed_distributions = pkg_resources.working_set

    # Extract distribution information and create a list of requirements
    requirements = [str(dist) for dist in installed_distributions]

    # Write the requirements to a requirements.txt file
    with open('requirements.txt', 'w') as req_file:
        req_file.write('\n'.join(requirements))

if __name__ == "__main__":
    main()
