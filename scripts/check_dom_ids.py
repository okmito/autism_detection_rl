import pathlib, re
t = pathlib.Path('scripts/demo_static/index.html').read_text(encoding='utf-8')
ids_defined = set(re.findall(r'id="([^"]+)"', t))
ids_used = set(re.findall(r"\$\('([a-zA-Z0-9_]+)'\)", t))
missing = sorted(i for i in ids_used if i not in ids_defined)
print('missing ids referenced by $():', missing)
print('all ids used:', sorted(ids_used))
