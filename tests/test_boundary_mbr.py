"""Boundary eligibility protects valid context and always retains a fallback."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('boundary',Path(__file__).resolve().parents[1]/'saul/serve/boundary.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


def test_boundary_choice():
    reasoning=' '.join(['The evidence needs further examination.']*8)
    restart=reasoning+' The appeal is dismissed. IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION'
    prompt='The parties dispute the interpretation of the provision.'
    result=b.boundary_choice([restart,reasoning],[.9,.8],prompt,0)
    assert result['selected_index']==1 and result['flagged']==[True,False]
    # A genuine heading or disposal already in the prompt exempts the request.
    for protected in ['IN THE SUPREME COURT OF INDIA', 'The appeal is dismissed.']:
        result=b.boundary_choice([restart,reasoning],[.9,.8],protected,0)
        assert result['selected_index']==0 and result['prompt_exempt']
    # A heading alone, an early continuation of one, and closure alone are allowed.
    for allowed in [reasoning+' CIVIL APPELLATE JURISDICTION',
                    'The appeal is dismissed. CIVIL APPELLATE JURISDICTION '+reasoning,
                    reasoning+' The appeal is dismissed.']:
        assert not b.boundary_choice([allowed],[1.],prompt,0)['flagged'][0]
    fallback=b.boundary_choice([restart,restart],[.8,.9],prompt,1)
    assert fallback['selected_index']==1 and fallback['fallback_all_flagged']
    # A native unflagged choice is preserved even if rounded scores tie.
    assert b.boundary_choice([reasoning,reasoning],[.8,.8],prompt,1)['selected_index']==1


if __name__=='__main__':
    test_boundary_choice()
    print('Boundary eligibility checks passed')
