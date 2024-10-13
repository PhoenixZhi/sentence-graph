import torch
from torch import nn
import numpy as np
from collections import defaultdict


class ISGModel(nn.Module):
    def __init__(self, word_emb,len_vocab,d_model=512,dropout=0.1):
        super(ISGModel, self).__init__()
        self.word_emb = word_emb
        self.len_vocab = len_vocab
        self.d_model = d_model

        self.drop_prob_lm = dropout

        self.embed2vis = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(self.drop_prob_lm))
        self.rel_sbj_fc=nn.Sequential(nn.Linear(self.d_model*3, self.d_model),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rel_obj_fc=nn.Sequential(nn.Linear(self.d_model*3, self.d_model),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rel_rel_fc = nn.Sequential(nn.Linear(self.d_model * 3, self.d_model),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(self.drop_prob_lm))
        self.rela_attr_fc = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
        self.layer_norm_region = nn.LayerNorm(self.d_model)
        self.device = torch.device('cuda')
    def forward(self,visual):
        attfeat=visual['reg_feat']
        isg_rel=visual['isg_rel']
        isg_obj=visual['isg_obj']
        isg_att=visual['isg_att']

        #make the mask
        N_batch, N_rel, _ = isg_rel.size()
        _, N_obj,N_attr = isg_att.size()
        rel_mask  = (isg_rel[:, :, 2] == 1)
        obj_mask  = (torch.sum(attfeat, dim=-1) == 0)
        #turn the word to vis embed
        rel_emb = self.word_emb(isg_rel[:, :, 2])
        rel_emb = self.embed2vis(rel_emb)
        attr_emb = self.word_emb(isg_att)
        attr_emb = self.embed2vis(attr_emb)

        rel_sbj_id = isg_rel[:,:,0]
        rel_obj_id = isg_rel[:,:,1]
        rel_sbj_feat=attfeat.gather(1,rel_sbj_id.unsqueeze(-1).repeat(1,1,self.d_model)).contiguous()
        rel_obj_feat=attfeat.gather(1,rel_obj_id.unsqueeze(-1).repeat(1,1,self.d_model)).contiguous()
        rel_sbj_feat=self.rel_sbj_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        rel_obj_feat=self.rel_obj_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        rel_rel_feat=self.rel_rel_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        attr_feat=self.rela_attr_fc( torch.cat((attr_emb,attfeat.unsqueeze(-2).repeat(1,1,N_attr,1) ),dim=-1) )

        # merg the feat of sbj and obj in rel
        attfeat=attfeat.scatter_add(1, rel_sbj_id.unsqueeze(-1).repeat(1,1, self.d_model), rel_sbj_feat)
        attfeat=attfeat.scatter_add(1, rel_obj_id.unsqueeze(-1).repeat(1,1, self.d_model), rel_obj_feat)
        rel_object_mask = torch.cat([rel_mask, rel_mask], dim=1)
        rel_object_id = torch.cat((rel_sbj_id, rel_obj_id), dim=1).masked_fill(rel_object_mask, N_obj)
        counts = torch.ones(N_batch, N_rel * 2, N_obj + 1).to(self.device)
        counts.scatter_(2, rel_object_id.unsqueeze(-1), 1)
        attfeat = attfeat / counts.sum(dim=1).unsqueeze(-1)[:, :N_obj]
        rel_rel_feat = (rel_rel_feat + rel_emb) / 2
        attr_feat = (attr_feat.sum(dim=2)) / 2
        attr_feat.masked_fill_(obj_mask.unsqueeze(-1).repeat(1, 1, self.d_model),0)
        #make the return
        new_visual = {}
        new_visual['reg_feat'] =torch.cat((attfeat, rel_rel_feat,attr_feat), dim=1)
        new_visual['reg_feat']=self.layer_norm_region(new_visual['reg_feat'])
        return new_visual
