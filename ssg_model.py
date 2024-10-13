import torch
from torch import nn




class SSGModel(nn.Module):
    def __init__(self, word_emb, d_model=512, dropout=0.1):
        super(SSGModel, self).__init__()

        self.drop_prob_lm = dropout
        self.d_model = d_model
        self.word_emb = word_emb
        self.rel_sbj_fc = nn.Sequential(nn.Linear(self.d_model * 3, self.d_model),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
        self.rel_obj_fc = nn.Sequential(nn.Linear(self.d_model * 3, self.d_model),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
        self.rel_rel_fc = nn.Sequential(nn.Linear(self.d_model * 3, self.d_model),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
        self.obj_fc = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_fc = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))


    def forward(self, visual):
        ssg_rel = visual['ssg_rel']
        ssg_obj = visual['ssg_obj']
        ssg_att = visual['ssg_att']
        N_batch, N_rel, _ = ssg_rel.size()
        _, N_obj, N_att = ssg_att.size()
        # turn the word to embedding
        rel_mask = (ssg_rel[:, :, 2] == 1)
        obj_mask = (ssg_obj == 1)
        attr_mask = (ssg_att == 1)
        rel_emb = self.word_emb(ssg_rel[:, :, 2])
        obj_emb = self.word_emb(ssg_obj)
        attr_emb = self.word_emb(ssg_att)

        # turn the embedding to model size
        rel_sbj_id = ssg_rel[:, :, 0]
        rel_obj_id = ssg_rel[:, :, 1]
        rel_sbj_feat = obj_emb.gather(1, rel_sbj_id.unsqueeze(-1).repeat(1,1,
                                                                         self.d_model)).contiguous()
        rel_obj_feat = obj_emb.gather(1, rel_obj_id.unsqueeze(-1).repeat(1,1,
                                                                         self.d_model)).contiguous()
        rel_sbj_feat = self.rel_sbj_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        rel_obj_feat = self.rel_obj_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        rel_rel_feat = self.rel_rel_fc(torch.cat([rel_sbj_feat, rel_obj_feat, rel_emb], dim=-1)).masked_fill(
            rel_mask.unsqueeze(-1), 0)
        obj_feat = self.obj_fc(obj_emb).masked_fill(obj_mask.unsqueeze(-1), 0)
        attr_feat = self.att_fc(torch.cat(
            [obj_emb.unsqueeze(-2).repeat(1,1, attr_emb.size(2), 1), attr_emb],
            dim=-1)).masked_fill(attr_mask.unsqueeze(-1), 0)

        # merg the feat of sbj and obj in rel
        obj_feat = obj_feat.scatter_add(1, rel_sbj_id.unsqueeze(-1).repeat(1,1, self.d_model),rel_sbj_feat)
        obj_feat = obj_feat.scatter_add(1, rel_obj_id.unsqueeze(-1).repeat(1,1, self.d_model),rel_obj_feat)
        rel_object_mask = torch.cat([rel_mask, rel_mask], dim=1)
        rel_object_id = torch.cat((rel_sbj_id, rel_obj_id), dim=1).masked_fill(rel_object_mask,N_obj)
        counts = torch.ones(N_batch, N_rel * 2, N_obj+1).cuda()
        counts.scatter_(2, rel_object_id.unsqueeze(-1), 1)
        rel_object_feat = obj_feat / counts.sum(dim=1).unsqueeze(-1)[:,:N_obj]

        # process attr feat and mask
        attr_feat = attr_feat.sum(dim=2)
        attr_mask = torch.sum(attr_mask == 0, dim=-1)
        attr_feat = attr_feat / attr_mask.unsqueeze(-1)
        attr_mask = (attr_mask == 0)
        attr_feat = attr_feat.masked_fill(attr_mask.unsqueeze(-1),0)
        # make the return
        ssg_feat = torch.cat([rel_object_feat, rel_rel_feat, attr_feat], dim=1)
        ssg_mask = torch.cat([obj_mask, rel_mask, attr_mask], dim=1).unsqueeze(-2).unsqueeze(-2)


        return ssg_feat,ssg_mask
