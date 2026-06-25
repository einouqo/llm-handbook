# Contributing

## Development environment

The project is set up to work with [`devenv`](https://devenv.sh/). Fill free to update the environment as needed.

To enter the environment:

```bash
devenv shell
```

## Available commands

Common commands are provided through the `Makefile`:

```bash
make html   # build the HTML site into site/
make pdf    # build the PDF into dist/
make clean  # remove generated artifacts
```

## Releases

- GitHub Pages builds and publishes the HTML site plus a PDF copy from `.github/workflows/pages.yml`.
- GitHub Releases publish `dist/llm-handbook.pdf` for semver-like tags from `.github/workflows/release.yml`.

## Contribution flow

1. Fork the repository.
2. Create a branch for your change.
3. Edit the handbook or supporting files.
4. Commit your changes.
5. Open a pull request.

Please include references when adding or revising handbook content whenever possible. Small fixes, clarifications, and corrections are welcome.
