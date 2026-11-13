import ast,json,pathlib
p=pathlib.Path(__file__).resolve().parent
files=['resume_gpu1_after_racnet.py','metadata_fixture.py']
for n in files: ast.parse((p/n).read_text(encoding='utf-8-sig'))
(p/'STATIC_PARSE.json').write_text(json.dumps(dict(status='PASS',files=files,source_only=True)))
print('STATIC_PARSE_PASS')
