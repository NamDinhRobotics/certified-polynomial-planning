"""Replay new physical records using separately implemented exact arithmetic."""
from pathlib import Path
import argparse,hashlib,json,sys,time
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import q1_independent as check

def verify(path,check_cells=True):
 data=json.loads(path.read_text());assert data['complete'];assert len({r['id'] for r in data['rows']})==len(data['rows']);curves=polys=cells=0;clearance=[]
 for row in data['rows']:
  arms=row.get('arms',{})
  for name,a in arms.items():
   if not a.get('ok'):continue
   # Numerical solver diagnostics are not physical outputs.
   if 'Gamma' not in a:continue
   v=check.curve(a,row['obstacles'],row['family']['k'],row['bc0'],row['bc1'],row['family']['eta'])
   curves+=1;polys+=v['polynomials'];clearance.append(dict(id=row['id'],arm=name,lower=str(v['squared_clearance_lower']),upper=str(v['squared_clearance_upper'])))
   if check_cells:
    for t in a.get('attempts',[]):
     if t['ok']:cells+=check.tail(t,row['obstacles'],row['bc0'],row['bc1'],row['family']['eta'])
 report=dict(passed=True,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(data['rows']),curves=curves,polynomials=polys,cells=cells,squared_clearance_brackets=clearance)
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.exists():raise FileExistsError(args.out)
 report=verify(args.input);args.out.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='squared_clearance_brackets'})
if __name__=='__main__':main()
