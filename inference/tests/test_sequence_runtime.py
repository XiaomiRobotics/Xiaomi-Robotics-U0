from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from xr_u0_ar.sequence_runtime import serialize_prompt, unconditional_context, generation_budget, save_sequence_parts
from xr_u0_ar.sequence_generation import generate_sequence, BOI_ID, IMG_ID, EOI_ID, EOL_ID, BSS_ID, ESS_ID, EOS_ID, VISUAL_TOKEN_OFFSET


class Tokenizer:
    bos_token_id = 151849
    special = {"<|extra_203|>": 151849, "<|extra_100|>": BSS_ID,
               "<|image start|>": BOI_ID, "<|image end|>": EOI_ID}

    def encode(self, text, **kwargs):
        ids = []
        while text:
            found = next((token for token in self.special if text.startswith(token)), None)
            if found:
                ids.append(self.special[found]); text = text[len(found):]
            else:
                ids.append(ord(text[0])); text = text[1:]
        return ids

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


class SequenceTests(unittest.TestCase):
    def test_sdpa_matches_eager_with_sparse_positions_and_cache(self):
        from xr_u0_ar.configuration_unis import UNISConfig
        from xr_u0_ar.modeling_unis import UNISForCausalLM
        torch.manual_seed(42)
        def model(attention):
            cfg = UNISConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, pad_token_id=0, attention_dropout=0.0)
            cfg._attn_implementation = attention
            return UNISForCausalLM(cfg).eval()
        eager, sdpa = model('eager'), model('sdpa')
        sdpa.load_state_dict(eager.state_dict())
        with torch.inference_mode():
            a=eager(input_ids=torch.tensor([[1,2,3]]),position_ids=torch.tensor([[0,7,12]]),use_cache=True)
            b=sdpa(input_ids=torch.tensor([[1,2,3]]),position_ids=torch.tensor([[0,7,12]]),use_cache=True)
            torch.testing.assert_close(a.logits,b.logits,atol=1e-5,rtol=1e-4)
            a=eager(input_ids=torch.tensor([[4]]),position_ids=torch.tensor([[14]]),past_key_values=a.past_key_values,use_cache=True)
            b=sdpa(input_ids=torch.tensor([[4]]),position_ids=torch.tensor([[14]]),past_key_values=b.past_key_values,use_cache=True)
            torch.testing.assert_close(a.logits,b.logits,atol=1e-5,rtol=1e-4)

    def test_prefix_preserves_whitespace_and_cfg_positions(self):
        tok = Tokenizer()
        prefix = '<|extra_203|>Task. Robot Arm Type: Test. Instruction: move.\n<|VIS_PLH|>\n<|extra_100|>'
        image = '<|image start|>1*1<|image end|>'
        ids = serialize_prompt(tok, prefix, [image])
        self.assertEqual(ids, tok.encode(prefix.replace('<|VIS_PLH|>', image)))
        uncond, positions = unconditional_context(tok, ids)
        expected = tok.encode('<|extra_203|>Task. Robot Arm Type: Test.' + image + '<|extra_100|>')
        self.assertEqual(uncond, expected)
        self.assertEqual(uncond, [ids[i] for i in positions])
        self.assertEqual(positions[-1], len(ids)-1)
        self.assertTrue(any(b-a > 1 for a,b in zip(positions, positions[1:])))
        self.assertEqual(generation_budget(tok, [[2, 3]], 10, 100), 15)
        self.assertEqual(generation_budget(tok, [[2, 3]], 10, 20), 10)

    def test_subtask_minimal_ordered_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = [{'kind':'text','text':'Step 1'}, {'kind':'image','frames':[Image.new('RGB',(16,16))]},
                     {'kind':'text','text':'Step 2'}, {'kind':'image','frames':[Image.new('RGB',(16,16))]}]
            path, count = save_sequence_parts(parts,tmp,'test','interleave_subtask',[],1)
            self.assertEqual(count,2)
            self.assertEqual([p['kind'] for p in json.loads(path.read_text())['parts']],['text','image','text','image'])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),['test.json','test_000.png','test_001.png'])

    def test_full_span_stops_and_preserves_sparse_cfg_positions(self):
        from types import SimpleNamespace
        tokens=[ord('A'),BOI_ID,ord('1'),ord('*'),ord('1'),IMG_ID,VISUAL_TOKEN_OFFSET,EOI_ID,ESS_ID,EOS_ID]
        class Model:
            def __init__(self): self.calls=[]
            def __call__(self, input_ids, position_ids, **kwargs):
                self.calls.append(position_ids.tolist()[0])
                position=int(position_ids[0,-1])
                index=0 if position < 5 else position-5+1
                logits=torch.full((1,1,VISUAL_TOKEN_OFFSET+1),-1000.0)
                logits[0,0,tokens[index]]=1000
                return SimpleNamespace(logits=logits,past_key_values=True)
        model=Model()
        result=generate_sequence(model,Tokenizer(),prefix_ids=torch.tensor([1,2,3,4,BSS_ID]),
            device=torch.device('cpu'),max_new_tokens=30,temperature=1,top_k=1,top_p=1,
            greedy=True,cfg_scale=3,uncond_context_ids=torch.tensor([1,BSS_ID]),
            uncond_context_position_ids=torch.tensor([0,4]),max_images=32,max_image_tokens=32768,
            max_header_tokens=32,control_greedy=True,bss_in_prefix=True,cfg_scale_end=1,cfg_decay_images=15)
        self.assertEqual(result['stop_reason'],'eos')
        self.assertTrue(result['format_valid'])
        self.assertEqual(result['generated_ids'],tokens)
        # The text token at absolute position 5 is fed only to the conditional cache.
        self.assertEqual(model.calls.count([5]),1)
        self.assertEqual(model.calls.count([6]),2)


if __name__ == '__main__':
    unittest.main()
