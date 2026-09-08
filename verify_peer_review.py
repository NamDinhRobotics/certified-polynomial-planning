"""Independently check all new curves and, with --full, new SDP intervals."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src'),str(ROOT/'verification')]
def run(full=False):
 import review_affine_green,certified_results
 prior=sys.argv
 try:
  sys.argv=['review_affine_green','--verify','--out',str(ROOT/'artifacts/review_affine_green.json')]
  review_affine_green.main()
 finally:sys.argv=prior
 curves=json.loads((ROOT/'artifacts/review_affine_green.verification.json').read_text())
 data=json.loads((ROOT/'artifacts/review_numpy_solver.json').read_text())
 assert data['protocol']['sos_polish_backend']=='NumPy; _C_POLISH=False enforced before construction'
 assert not certified_results.validate(data)
 result={'curves':curves,'numpy_rows':len(data['rows']),'backend':data['protocol']['sos_polish_backend']}
 if full:result['numpy_exact']=certified_results.recheck(data)
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
 print(json.dumps(run(args.full),indent=2))
