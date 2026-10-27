"""Read-only publication integrity check; never invokes an optimizer."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,zipfile,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def verify():
    manifest=read(ROOT/'PUBLIC_MANIFEST.json')
    assert len({r['path'] for r in manifest})==len(manifest)
    for row in manifest:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT),row['path']
        assert sha(path)==row['sha256'],row['path']
    source=ROOT/'reference_results/confirmation_20261024'
    plan=read(source/'plan.json');done=read(source/'DONE.json')
    assert done['status']=='complete' and done['runs']==180 and done['independent_audits_passed']
    assert read(source/'analysis/DONE.json')['status']=='complete'
    for relative,digest in plan['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
    assert len(plan['jobs'])==180 and len({j['id'] for j in plan['jobs']})==180
    audits={}
    geometry=0
    for case in plan['cases']:
        audit=read(source/'analysis'/case['id']/'independent_audit.json')
        assert audit['status']=='passed';geometry+=audit['geometry_records']
        audits.update({r['id']:r for r in audit['rows']})
        assert read(source/'analysis'/case['id']/'full_objective_rechecks.json')['status']=='passed'
    for job in plan['jobs']:
        folder=source/'runs'/job['id']
        record=read(folder/'DONE.json')
        assert record['status']=='complete' and record['requests']==10000
        assert record['config']['seed']==job['seed'] and record['method']==job['method']
        assert sha(source/'configs'/(job['id']+'.json'))==job['config_sha256']
        assert sha(source/'jobs'/(job['id']+'.json'))==job['worker_job_sha256']
        for name,digest in audits[job['id']]['hashes'].items():assert sha(folder/name)==digest,(job['id'],name)
        assert (folder/'case_metadata.json').exists()
    assert geometry==1800000
    assert len(rows(source/'analysis/summary.csv'))==18
    assert len(rows(source/'analysis/all_runs.csv'))==180
    assert len(rows(source/'analysis/all_paired.csv'))==90
    assert len(rows(source/'analysis/primary_tests.csv'))==18
    metadata=read(ROOT/'reports/REPORT_METADATA.json')
    report=ROOT/metadata['repository_path']
    assert sha(report)==metadata['sha256']
    with zipfile.ZipFile(report) as z:
        tree=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text=''.join(t.text or '' for t in tree.findall('.//w:t',ns))
        assert '成员分工' in text
        labels=metadata['member_labels']
        assert len(labels)==metadata['members']==7 and len(set(labels))==7
        assert all(label in text for label in labels)
    handbook=read(ROOT/'reports/HANDBOOK_METADATA.json')
    assert sha(ROOT/handbook['repository_path'])==handbook['sha256']
    assert handbook['member_labels']==metadata['member_labels']
    return dict(status='passed',finished_at=datetime.datetime.now().astimezone().isoformat(),
        manifest_files=len(manifest),frozen_source_fingerprints_matched=True,complete_runs=180,
        original_audit_hashes_matched=True,geometry_records_in_verified_original_audits=geometry,
        report_sha256=metadata['sha256'],report_members=7,report_pages=metadata['pages'],
        handbook_sha256=handbook['sha256'],handbook_pages=handbook['pages'],
        optimization_runs_started=0,
        scope='Published bytes and saved audit evidence verified; no new model evaluation or performance experiment')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    result=verify()
    if args.output:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
