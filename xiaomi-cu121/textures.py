from pathlib import Path
from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY, download_and_extract_zip
spec=dict(DOWNLOAD_ASSET_REGISTRY['tex_generative'])
spec['folder']='/work/extra-assets/generative_textures'
download_and_extract_zip(**spec)
assert Path('/work/extra-assets/generative_textures/wall/tex071.png').is_file()
print('REQUIRED_TEXTURE_PRESENT',flush=True)
