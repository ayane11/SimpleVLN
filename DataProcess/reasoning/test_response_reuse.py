import unittest,tempfile,json
from pathlib import Path
from response_reuse import load_responses,request_identity
class ResponseReuseTest(unittest.TestCase):
 def test_error_log_uses_response_for_last_stored_prompt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'errors').mkdir();s={'model':'same'};m=[{'role':'user','content':'repair feedback'}]
   (p/'errors/x.json').write_text(json.dumps(dict(settings=s,messages=m,attempts=[{'response':{'text':'old response'}},{'response':{'text':'last response'}}])))
   cache=load_responses(p,s)
   self.assertEqual(cache[request_identity(m,[],s)]['text'],'last response')
 def test_validator_hash_can_change_but_model_inputs_cannot(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'requests').mkdir();s={'model':'same','max_new_tokens':100,'code_hash':'old'}
   m=[{'role':'user','content':'example'}];im=[]
   (p/'requests/x.json').write_text(json.dumps(dict(settings=s,messages=m,image_files=im,response=dict(text='{}',seconds=3))))
   new=dict(s,code_hash='new');cache=load_responses(p,new)
   value=cache[request_identity(m,im,new)];self.assertEqual(value['text'],'{}');self.assertTrue(value['reused_from']);self.assertEqual(value['seconds'],0)
   self.assertNotIn(request_identity([{'role':'user','content':'changed'}],im,new),cache)
   self.assertNotIn(request_identity(m,[{'path':'changed','mtime_ns':1}],new),cache)
   self.assertFalse(load_responses(p,dict(new,max_new_tokens=101)))
if __name__=='__main__':unittest.main()
