import marimo

app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # pytest Tutorial: Cosmographic Analysis

        A hands-on walkthrough using functions from your own codebase.

        **Run cells in order.** Each phase ends with a test run so you see the output immediately.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Phase 1: The "Zero Setup" Basics

        **Idea:** Pure functions don't need mocks, fixtures, or setup — just `assert` and run.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        We test `cosmology.dl(z, h0, q0)` and `cosmology.mu(z, h0, q0)` from
        `src/cosmographic_analysis/cosmology.py`:

        ```python
        def dl(z, h0, q0):
            y = z / (z + 1.0)
            return (2997.92458 / h0) * (y + (3.0 - q0) * y * y / 2.0)

        def mu(z, h0, q0):
            return 5.0 * np.log10(dl(z, h0, q0)) + 25.0
        ```

        Tests check invariants: zero z → zero distance, monotonicity, inverse-h0 scaling.
        There's also one **intentional failure** so you can see pytest's diff output.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("**Run Phase 1 tests:**")
    return


@app.cell
def __():
    import subprocess, os

    result = subprocess.run(
        [
            "conda", "run", "-n", "cosmographic_analysis",
            "python", "-m", "pytest",
            "pytest_tutorial/test_phase1_basics.py", "-v",
        ],
        capture_output=True,
        text=True,
        cwd="/home/diego/Projects/cosmographic_analysis",
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return


@app.cell
def __(mo):
    mo.md(
        """
        **What to look for:**
        - 4 green `PASSED` — those work
        - 1 red `FAILED` — the intentional one. Notice how pytest prints **both** the expected and actual value on the `assert` line.
        - The error shows `assert 212.20050136507936 == 999.0` — instant feedback, no `print()` debugging needed.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Phase 2: Parameterization — One Test, Dozens of Cases

        **Idea:** `@pytest.mark.parametrize` lets you run the same test logic against many inputs.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        We test `coordinates.DecRa2Cartesian(dec, ra)` from
        `src/cosmographic_analysis/coordinates.py:25-49`.

        Without parametrize you'd write 6 nearly identical functions. With it:

        ```python
        @pytest.mark.parametrize("dec, ra, exp_x, exp_y, exp_z", [
            (90.0,   0.0,  0.0,  0.0,  1.0),   # North pole
            (-90.0,  0.0,  0.0,  0.0, -1.0),   # South pole
            (0.0,    0.0,  1.0,  0.0,  0.0),   # Equator, RA=0
            (0.0,   90.0,  0.0,  1.0,  0.0),   # Equator, RA=90
            (0.0,  180.0, -1.0,  0.0,  0.0),   # Equator, RA=180
            (0.0,  -90.0,  0.0, -1.0,  0.0),   # Negative angle edge case
        ])
        def test_DecRa2Cartesian_known_points(dec, ra, exp_x, exp_y, exp_z):
            result = DecRa2Cartesian(np.array([dec]), np.array([ra]))
            assert np.isclose(result[0][0], exp_x)
            assert np.isclose(result[0][1], exp_y)
            assert np.isclose(result[0][2], exp_z)
        ```

        pytest generates one test **per row** — if row 3 fails but the rest pass, you see exactly which inputs caused it.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("**Run Phase 2 tests:**")
    return


@app.cell
def __():
    import subprocess, os

    result = subprocess.run(
        [
            "conda", "run", "-n", "cosmographic_analysis",
            "python", "-m", "pytest",
            "pytest_tutorial/test_phase2_parametrize.py", "-v",
        ],
        capture_output=True,
        text=True,
        cwd="/home/diego/Projects/cosmographic_analysis",
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return


@app.cell
def __(mo):
    mo.md(
        """
        **What to look for:**
        - Each row in `@parametrize` appears as a separate test:
          `test_DecRa2Cartesian_known_points[90.0-0.0-0.0-0.0-1.0]`
        - The bracket notation tells you exactly which parameters were used
        - 9 tests from 2 functions — no code duplication
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Phase 3: Fixtures — Clean Setup Without Boilerplate

        **Idea:** Fixtures replace repeated `setup()` / `teardown()` boilerplate with dependency injection.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        The `Config` dataclass in `config.py` needs to be constructed with nested dataclasses.
        Instead of writing this in every test:

        ```python
        def test_something():
            cfg = Config(parameters=ParametersConfig(...), ...)
            ...
        ```

        Define the fixture **once** in `conftest.py`:

        ```python
        @pytest.fixture
        def default_config():
            return Config()

        @pytest.fixture
        def custom_config():
            return Config(parameters=ParametersConfig(nside=32, ...))
        ```

        Tests just declare the parameter — pytest injects it.
        Each test gets a **fresh** copy, so mutations don't leak between tests.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("**Run Phase 3 tests:**")
    return


@app.cell
def __():
    import subprocess, os

    result = subprocess.run(
        [
            "conda", "run", "-n", "cosmographic_analysis",
            "python", "-m", "pytest",
            "pytest_tutorial/test_phase3_fixtures.py", "-v",
        ],
        capture_output=True,
        text=True,
        cwd="/home/diego/Projects/cosmographic_analysis",
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return


@app.cell
def __(mo):
    mo.md(
        """
        **What to look for:**
        - Tests grouped in classes (`TestConfigDefaults`, `TestCustomConfig`)
        - `test_modifications_do_not_leak_across_tests` sets `h0f = 999.0`
        - `test_config_is_unchanged_by_previous_test` still sees `h0f == 0.7304` — proving isolation
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Phase 4: Mocking — Test Without Touching the Real World

        **Idea:** Code that reads files or calls APIs is slow and brittle. Mocking replaces those calls with fake data.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        We test `config.load_config()` which normally opens `config.yaml` and parses it.
        Using `mocker` (from `pytest-mock`), we intercept the file I/O:

        ```python
        def test_load_config_with_mocked_yaml(mocker):
            fake_data = {"parameters": {"nside": 64, "h0f": 0.68}}
            mocker.patch("cosmographic_analysis.config.yaml.safe_load",
                         return_value=fake_data)
            mocker.patch.object(Path, "exists", return_value=True)
            mocker.patch("builtins.open", mocker.mock_open())

            config = load_config("fake_path.yaml")
            assert config.parameters.nside == 64
        ```

        Three scenarios:
        1. **Happy path** — mock returns custom YAML data
        2. **Missing file** — `Path.exists() \\u2192 False`, verify defaults are used
        3. **Empty YAML** — `safe_load \\u2192 None`, verify graceful fallback
        """
    )
    return


@app.cell
def __(mo):
    mo.md("**Run Phase 4 tests:**")
    return


@app.cell
def __():
    import subprocess, os

    result = subprocess.run(
        [
            "conda", "run", "-n", "cosmographic_analysis",
            "python", "-m", "pytest",
            "pytest_tutorial/test_phase4_mocking.py", "-v",
        ],
        capture_output=True,
        text=True,
        cwd="/home/diego/Projects/cosmographic_analysis",
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return


@app.cell
def __(mo):
    mo.md(
        """
        **What to look for:**
        - No real YAML file was created — the tests never touch disk
        - `test_load_config_fallback_to_defaults` proves graceful handling of missing files
        - All 3 tests complete in milliseconds
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Summary

        | Phase | Tool | What it solved |
        |---|---|---|
        | 1 | `assert` | Proves testing can be zero-setup for pure functions |
        | 2 | `@parametrize` | Tests 9+ cases without code duplication |
        | 3 | `@pytest.fixture` | Separates setup logic from test logic |
        | 4 | `mocker.patch` | Tests file-reading code without real files |

        **Try extending it:** Pick any other function in `src/cosmographic_analysis/` and write a test for it using one of these patterns.
        """
    )
    return


if __name__ == "__main__":
    app.run()
