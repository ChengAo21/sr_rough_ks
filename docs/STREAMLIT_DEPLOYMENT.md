# Deploy Roughness Lab on Streamlit

Application URL: **https://jfe-rough-sr.streamlit.app/**

## Deployment settings

Use these settings in Streamlit Community Cloud:

| Setting | Value |
|---|---|
| Repository | `ChengAo21/sr_rough_ks` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| App URL | `jfe-rough-sr` |
| Python version | **3.12** |

Python 3.12 matches the locally verified runtime. Select it under **Advanced settings → Python version**. No secrets are required for the base application.

## Publish or update

1. Extract the complete release archive.
2. Copy the **contents** of its `JFE_rough_codes` folder into the root of the GitHub repository. Keep hidden files such as `.streamlit/config.toml` and `.gitignore`.
3. Commit and push the updated files to `main`. The repository root should contain `streamlit_app.py`, `requirements.txt`, `roughlab/`, `code_files/`, `results/`, `assets/` and `docs/`.
4. In Community Cloud, use the settings above and select **Deploy**. If the app is already deployed from this branch, pushing the update triggers a rebuild or reload.
5. Open the app URL and confirm that Overview, Symbolic Lab and PCE Studio load. Run a small Python GP search and a PCE fit to verify the application after deployment.

The repository's `requirements.txt` is installed automatically. The base deployment supports live Python GP searches, PCE calculations, the model explorer and the saved PySR experiment. Live PySR searches additionally require the optional dependencies in `requirements-pysr.txt` and Julia initialization; local setup is documented in [LOCAL_DEMO.md](LOCAL_DEMO.md).

The theme and headless settings are in `.streamlit/config.toml`. The cloud platform chooses the server bind address; the Windows launcher explicitly binds local previews to `127.0.0.1`.

## Official documentation

- [Deploy an app and choose the Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
- [Declare application dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
