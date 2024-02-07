"""Models for Text and Image Composition."""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from torch.autograd import Variable
from transformer import MultiHeadAttentionOne


def reshape_text_features(text_features, image_features_shapes):
    return text_features.view((*text_features.size(), 1, 1)).repeat(1, 1, *image_features_shapes[2:])
    
def calculate_img_mean_std(x):
    mu_x = torch.mean(x, dim=(2, 3), keepdim=True).detach()
    std_x = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False).detach()
    return mu_x, std_x
    
def calculate_mean_std(t):
    mu_t = torch.mean(t).detach()
    std_t = torch.std(t).detach()
    return mu_t, std_t
    
class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, 1)

        return x

class NonLocalBlock(torch.nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = torch.nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = torch.nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = torch.nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv_mask = torch.nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        #print ('b c h w', b, c, h, w)
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out        

class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()
        self.name = 'model_name'

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert):
        raise NotImplementedError

    def compose_img_text(self, imgs, masks, text_query):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query,
                     imgs_query_mask,
                     text_query,
                     imgs_target,
                     soft_triplet_loss):
        dct_with_representations = self.compose_img_text(imgs_query, imgs_query_mask, text_query)
        composed_source_image = self.normalization_layer(dct_with_representations["repres"])
        composed_source_imageL = self.normalization_layer(dct_with_representations["Llayer_repres"])
        composed_source_imageM = self.normalization_layer(dct_with_representations["Mlayer_repres"])
        composed_source_imageH = self.normalization_layer(dct_with_representations["Hlayer_repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm["repres"])
        target_img_featuresL = self.normalization_layer(target_img_features_non_norm["Llayer_repres"])
        target_img_featuresM = self.normalization_layer(target_img_features_non_norm["Mlayer_repres"])
        target_img_featuresH = self.normalization_layer(target_img_features_non_norm["Hlayer_repres"])
        assert (composed_source_image.shape[0] == target_img_features.shape[0] and
                composed_source_image.shape[1] == target_img_features.shape[1])
        # Get Rot_Sym_Loss
        if self.name == 'composeAE':
            CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
            conjugate_representations = self.compose_img_text_features(target_img_features_non_norm, dct_with_representations["text_features"], CONJUGATE)
            composed_target_image = self.normalization_layer(conjugate_representations["repres"])
            source_img_features = self.normalization_layer(dct_with_representations["img_features"]) #img1
            if soft_triplet_loss:
                dct_with_representations ["rot_sym_loss"]= \
                    self.compute_soft_triplet_loss_(composed_target_image,source_img_features)
            else:
                dct_with_representations ["rot_sym_loss"]= \
                    self.compute_batch_based_classification_loss_(composed_target_image,
                                                              source_img_features)
        else: # tirg, RealSpaceConcatAE etc
            dct_with_representations ["rot_sym_loss"] = 0

        if soft_triplet_loss:
            loss0 = self.compute_soft_triplet_loss_(composed_source_image , target_img_features)
            lossL = self.compute_soft_triplet_loss_(composed_source_imageL , target_img_featuresL)
            lossM = self.compute_soft_triplet_loss_(composed_source_imageM , target_img_featuresM)
            lossH = self.compute_soft_triplet_loss_(composed_source_imageH , target_img_featuresH)
            all_loss = loss0 + lossL +  lossM +  lossH
            return all_loss, dct_with_representations
        else:
            loss0 = self.compute_batch_based_classification_loss_(composed_source_image , target_img_features)
            lossL = self.compute_batch_based_classification_loss_(composed_source_imageL , target_img_featuresL)
            lossM = self.compute_batch_based_classification_loss_(composed_source_imageM , target_img_featuresM)
            lossH = self.compute_batch_based_classification_loss_(composed_source_imageH , target_img_featuresH)
            all_loss = loss0 + lossL +  lossM +  lossH
            return all_loss, dct_with_representations
            #return self.compute_batch_based_classification_loss_(composed_source_image,
                                                                 #target_img_features), dct_with_representations

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        soft_triplet_loss = self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)
        reconstruction_loss = (mod_img1 - img2)**2
        #return soft_triplet_loss + 0.001*reconstruction_loss.mean().cpu()
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        #y = 1. -log_softmax(x, 1)
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        cross_entropy = F.cross_entropy(x, labels)
        #reconstruction_loss = (mod_img1 - img2)**2
        #similarity_loss = nn.KLDivLoss()(nn.LogSoftmax()(mod_img1), nn.Softmax()(img2))
        return cross_entropy #+ similarity_loss
        #return F.nll_loss(y, labels)

class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__()
        # img model
        img_model = torchvision.models.resnet18(pretrained=True)
        self.name = name

        class GlobalAvgPool2d(torch.nn.Module):

            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, image_embed_dim))
        img_model.fc1 = torch.nn.Sequential(torch.nn.Linear(64+128+512, image_embed_dim))
        self.img_model = img_model

        # text model
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab = text_query,
            word_embed_dim = text_embed_dim,
            lstm_hidden_dim = text_embed_dim)
        
        self.outL = torch.nn.Sequential(
            torch.nn.Conv2d(2 * 64, 64, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.outM = torch.nn.Sequential(
            torch.nn.Conv2d(2 * 128, 128, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.outH = torch.nn.Sequential(
            torch.nn.Conv2d(2 * 512, 512, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )        
        
    '''def extract_img_feature(self, imgs):
        return self.img_model(imgs)'''

    def extract_img_feature(self, imgs):
        #return self.img_model(imgs)
        x = imgs
        x = self.img_model.conv1(x)
        x = self.img_model.bn1(x)
        x = self.img_model.relu(x)
        x = self.img_model.maxpool(x)
        #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
        L_layer_imgfeature = x
        #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

        x = self.img_model.layer1(x)
        x = self.img_model.layer2(x)
        #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
        M_layer_imgfeature = x
        x = self.img_model.layer3(x)
        x = self.img_model.layer4(x)
        #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
        H_layer_imgfeature = x
        L_layer_imgfeature = self.img_model.avgpool(L_layer_imgfeature)
        #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)
        M_layer_imgfeature = self.img_model.avgpool(M_layer_imgfeature)
        H_layer_imgfeature = self.img_model.avgpool(H_layer_imgfeature)
        ALL_layer_imgfeature = torch.cat((L_layer_imgfeature, M_layer_imgfeature, H_layer_imgfeature), dim=1)
        #print ('ALL_layer_imgfeature shape:' ,ALL_layer_imgfeature.shape)
        #x = self.img_model.avgpool(ALL_layer_imgfeature)
        x = ALL_layer_imgfeature
        #print ('x shape:' ,x.shape)
        x = x.view(x.size(0), -1)
        #print ('x shape:' ,x.shape)
        x = self.img_model.fc1(x)
        L_layer_imgfeature = L_layer_imgfeature.view(L_layer_imgfeature.size(0), -1)
        M_layer_imgfeature = M_layer_imgfeature.view(M_layer_imgfeature.size(0), -1)
        H_layer_imgfeature = H_layer_imgfeature.view(H_layer_imgfeature.size(0), -1)
        representations = {"repres": x,
                   "Llayer_repres": L_layer_imgfeature,
                   "Mlayer_repres": M_layer_imgfeature,
                   "Hlayer_repres": H_layer_imgfeature
                   }
        #print ('x shape:' ,x.shape)
        return representations


    def extract_text_feature(self, text_query, use_bert):
        '''if use_bert:
            text_features = bc.encode(text_query)# obtain the text features
            return torch.from_numpy(text_features).cuda()'''
        return self.text_model(text_query)

class ourwMask(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
       
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    L_layer_imgfeature = x


    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    M_layer_imgfeature = x
    
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    L_layer_mskfeature = x

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    M_layer_mskfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    H_layer_mskfeature = x
    
    # images and masks feature
    L_layer_imgfeaturec = self.outL(torch.cat([L_layer_imgfeature, L_layer_mskfeature], dim=1))
    M_layer_imgfeaturec = self.outM(torch.cat([M_layer_imgfeature, M_layer_mskfeature], dim=1))
    H_layer_imgfeaturec = self.outH(torch.cat([H_layer_imgfeature, H_layer_mskfeature], dim=1))

    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.reshape(L_layer_imgfeaturec.shape)
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.reshape(M_layer_imgfeaturec.shape)
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.reshape(H_layer_imgfeaturec.shape)
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)

    L_layer_copfeature = L_layer_imgfeaturec * L_layer_txtfeature
    M_layer_copfeature = M_layer_imgfeaturec * M_layer_txtfeature
    H_layer_copfeature = H_layer_imgfeaturec * H_layer_txtfeature

    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations
    
class ourwMask1(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_mskfeature = x
    #print ('L_layer_mskfeature shape:' ,L_layer_mskfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_mskfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_mskfeature = x
    
    # images and masks feature
    L_layer_imgfeaturec = self.outL(torch.cat([L_layer_imgfeature, L_layer_mskfeature], dim=1))
    M_layer_imgfeaturec = self.outM(torch.cat([M_layer_imgfeature, M_layer_mskfeature], dim=1))
    H_layer_imgfeaturec = self.outH(torch.cat([H_layer_imgfeature, H_layer_mskfeature], dim=1))

    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.reshape(L_layer_imgfeaturec.shape)
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.reshape(M_layer_imgfeaturec.shape)
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.reshape(H_layer_imgfeaturec.shape)
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)

       
    L_layer_copfeature = NonLocalBlock(channel=64).cuda()(L_layer_imgfeaturec * L_layer_txtfeature)
    M_layer_copfeature = M_layer_imgfeaturec * M_layer_txtfeature #NonLocalBlock(channel=128).cuda()(M_layer_imgfeaturec * M_layer_txtfeature)
    H_layer_copfeature = NonLocalBlock(channel=512).cuda()(H_layer_imgfeaturec * H_layer_txtfeature)

    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations    

class ourwMask2(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_mskfeature = x
    #print ('L_layer_mskfeature shape:' ,L_layer_mskfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_mskfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_mskfeature = x

    #texts feature
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.reshape(L_layer_imgfeature.shape)
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.reshape(M_layer_imgfeature.shape)
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.reshape(H_layer_imgfeature.shape)
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)
    
    # texts and masks feature
    L_layer_modfeature = self.outL(torch.cat([L_layer_txtfeature, L_layer_mskfeature], dim=1))
    M_layer_modfeature = self.outM(torch.cat([M_layer_txtfeature, M_layer_mskfeature], dim=1))
    H_layer_modfeature = self.outH(torch.cat([H_layer_txtfeature, H_layer_mskfeature], dim=1))
       
    L_layer_copfeature = NonLocalBlock(channel=64).cuda()(L_layer_modfeature * L_layer_imgfeature)
    M_layer_copfeature = M_layer_modfeature * M_layer_imgfeature
    H_layer_copfeature = NonLocalBlock(channel=512).cuda()(H_layer_modfeature * H_layer_imgfeature)

    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations   

class ourwMask3(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    self.L_transformer = MultiHeadAttentionOne(4, 64, 64, 64, dropout=0.5)
    self.M_transformer = MultiHeadAttentionOne(4, 128, 128, 128, dropout=0.5)
    self.H_transformer = MultiHeadAttentionOne(4, 512, 512, 512, dropout=0.5)
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30*45),
             #torch.nn.ReLU(),
             #torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15*23),
             #torch.nn.ReLU(),
             #torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4*6),
             #torch.nn.ReLU(),
             #torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_mskfeature = x
    #print ('L_layer_mskfeature shape:' ,L_layer_mskfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_mskfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_mskfeature = x

    #texts feature
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.view(L_layer_imgfeature.size()[0], L_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.view(M_layer_imgfeature.size()[0], M_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.view(H_layer_imgfeature.size()[0], H_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)
    
    #co_ feature   
    L_layer_copfeature = self.L_transformer(L_layer_txtfeature, L_layer_mskfeature, L_layer_imgfeature).permute(0, 2, 1).contiguous().view(L_layer_imgfeature.size())
    M_layer_copfeature = self.M_transformer(M_layer_txtfeature, M_layer_mskfeature, M_layer_imgfeature).permute(0, 2, 1).contiguous().view(M_layer_imgfeature.size())
    H_layer_copfeature = self.H_transformer(H_layer_txtfeature, H_layer_mskfeature, H_layer_imgfeature).permute(0, 2, 1).contiguous().view(H_layer_imgfeature.size())
    #print ('L_layer_imgfeature shape:' ,L_layer_copfeature.shape)


    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations
    
class ourwMask4(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    self.L_transformer = MultiHeadAttentionOne(4, 64, 64, 64, dropout=0.1)
    self.M_transformer = MultiHeadAttentionOne(4, 128, 128, 128, dropout=0.1)
    self.H_transformer = MultiHeadAttentionOne(4, 512, 512, 512, dropout=0.1)
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30*45),
             #torch.nn.ReLU(),
             #torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15*23),
             #torch.nn.ReLU(),
             #torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4*6),
             #torch.nn.ReLU(),
             #torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_mskfeature = x
    #print ('L_layer_mskfeature shape:' ,L_layer_mskfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_mskfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_mskfeature = x

    #texts feature
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.view(L_layer_imgfeature.size())
    L_layer_txtfeature = L_layer_txtfeature*L_layer_mskfeature
    L_layer_txtfeature = L_layer_txtfeature.view(L_layer_imgfeature.size()[0], L_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.view(M_layer_imgfeature.size())
    M_layer_txtfeature = M_layer_txtfeature*M_layer_mskfeature    
    M_layer_txtfeature = M_layer_txtfeature.view(M_layer_imgfeature.size()[0], M_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.view(H_layer_imgfeature.size())
    H_layer_txtfeature = H_layer_txtfeature*H_layer_mskfeature    
    H_layer_txtfeature = H_layer_txtfeature.view(H_layer_imgfeature.size()[0], H_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)
    
    #co_ feature   
    L_layer_copfeature = self.L_transformer(L_layer_txtfeature, L_layer_imgfeature, L_layer_imgfeature).permute(0, 2, 1).contiguous().view(L_layer_imgfeature.size())
    M_layer_copfeature = self.M_transformer(M_layer_txtfeature, M_layer_imgfeature, M_layer_imgfeature).permute(0, 2, 1).contiguous().view(M_layer_imgfeature.size())
    H_layer_copfeature = self.H_transformer(H_layer_txtfeature, H_layer_imgfeature, H_layer_imgfeature).permute(0, 2, 1).contiguous().view(H_layer_imgfeature.size())
    #print ('L_layer_imgfeature shape:' ,L_layer_copfeature.shape)


    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations    

class ourwMask5(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    self.L_transformer = MultiHeadAttentionOne(4, 64, 64, 64, dropout=0.5)
    self.M_transformer = MultiHeadAttentionOne(4, 128, 128, 128, dropout=0.5)
    self.H_transformer = MultiHeadAttentionOne(4, 512, 512, 512, dropout=0.5)
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x

    #texts feature
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.view(L_layer_imgfeature.size()[0], L_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.view(M_layer_imgfeature.size()[0], M_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)    
    H_layer_txtfeature = H_layer_txtfeature.view(H_layer_imgfeature.size()[0], H_layer_imgfeature.size()[1], -1).permute(0, 2, 1).contiguous()
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)
    
    #co_ feature   
    L_layer_copfeature = self.L_transformer(L_layer_txtfeature, L_layer_imgfeature, L_layer_imgfeature).permute(0, 2, 1).contiguous().view(L_layer_imgfeature.size())
    M_layer_copfeature = self.M_transformer(M_layer_txtfeature, M_layer_imgfeature, M_layer_imgfeature).permute(0, 2, 1).contiguous().view(M_layer_imgfeature.size())
    H_layer_copfeature = self.H_transformer(H_layer_txtfeature, H_layer_imgfeature, H_layer_imgfeature).permute(0, 2, 1).contiguous().view(H_layer_imgfeature.size())
    #print ('L_layer_imgfeature shape:' ,L_layer_copfeature.shape)


    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations 
    
class ourwMask6(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #images feature
    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    #masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    L_layer_mskfeature = x

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    M_layer_mskfeature = x
    
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    H_layer_mskfeature = x
    
    # images and masks feature
    L_layer_imgfeaturec = 1e-5 * L_layer_imgfeature * L_layer_mskfeature + L_layer_imgfeature
    M_layer_imgfeaturec = 1e-5 * M_layer_imgfeature * M_layer_mskfeature + M_layer_imgfeature
    H_layer_imgfeaturec = 1e-5 * H_layer_imgfeature * H_layer_mskfeature + H_layer_imgfeature
 
    #text feature 
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.reshape(L_layer_imgfeature.shape)
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.reshape(M_layer_imgfeature.shape)
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.reshape(H_layer_imgfeature.shape)
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)

    mu_xL, std_xL = calculate_img_mean_std(L_layer_imgfeature)
    mu_xM, std_xM = calculate_img_mean_std(M_layer_imgfeature)
    mu_xH, std_xH = calculate_img_mean_std(H_layer_imgfeature)
    
    L_layer_copfeature = ((L_layer_imgfeaturec - mu_xL)/(std_xL + 1e-5)) * L_layer_txtfeature + L_layer_imgfeature
    M_layer_copfeature = ((M_layer_imgfeaturec - mu_xM)/(std_xM + 1e-5)) * M_layer_txtfeature + M_layer_imgfeature
    H_layer_copfeature = ((H_layer_imgfeaturec - mu_xH)/(std_xH + 1e-5)) * H_layer_txtfeature + H_layer_imgfeature


    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations

class ourwMask7(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, dataset):
    super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, dataset)
    self.use_bert = use_bert
    
    if  dataset == 'css3d':
        self.textLlayer = torch.nn.Sequential(           
             torch.nn.Linear(text_embed_dim, 64*30),
             torch.nn.ReLU(),
             torch.nn.Linear(64*30, 64*30*45),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*15),
             torch.nn.ReLU(),
             torch.nn.Linear(128*15, 128*15*23),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*4),
             torch.nn.ReLU(),
             torch.nn.Linear(512*4, 512*4*6),
        )        
        
    elif dataset == 'fashion200k':
        self.textLlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 64*56*56),
        )     

        self.textMlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 128*28*28),
        ) 
        
        self.textHlayer = torch.nn.Sequential(
             torch.nn.Linear(text_embed_dim, 512*7*7),
        )  
        #self.clsssifier = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, 9853))
    
  def compose_img_text(self, imgs, masks, text_query):
    text_features = self.extract_text_feature(text_query, self.use_bert)
    #print('img:', imgs[0], 'masks', masks[0])
    #images feature
    x = imgs + masks * imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    L_layer_imgfeature = x
    #print ('L_layer_imgfeature shape:' ,L_layer_imgfeature.shape)

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    M_layer_imgfeature = x
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    H_layer_imgfeature = x
    
    '''#masks feature
    x = masks
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    L_layer_mskfeature = x

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    M_layer_mskfeature = x
    
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    H_layer_mskfeature = x
    
    # images and masks feature
    L_layer_imgfeature = L_layer_imgfeature * L_layer_mskfeature + L_layer_imgfeature
    M_layer_imgfeature = M_layer_imgfeature * M_layer_mskfeature + M_layer_imgfeature
    H_layer_imgfeature = H_layer_imgfeature * H_layer_mskfeature + H_layer_imgfeature'''
 
    #text feature 
    L_layer_txtfeature = self.textLlayer(text_features)
    L_layer_txtfeature = L_layer_txtfeature.reshape(L_layer_imgfeature.shape)
    #L_layer_txtfeature = F.tanh(L_layer_txtfeature)

    M_layer_txtfeature = self.textMlayer(text_features)
    M_layer_txtfeature = M_layer_txtfeature.reshape(M_layer_imgfeature.shape)
    #M_layer_txtfeature = F.tanh(M_layer_txtfeature)

    H_layer_txtfeature = self.textHlayer(text_features)
    H_layer_txtfeature = H_layer_txtfeature.reshape(H_layer_imgfeature.shape)
    #H_layer_txtfeature = F.tanh(H_layer_txtfeature)

    mu_xL, std_xL = calculate_img_mean_std(L_layer_imgfeature)
    mu_xM, std_xM = calculate_img_mean_std(M_layer_imgfeature)
    mu_xH, std_xH = calculate_img_mean_std(H_layer_imgfeature)
    
    L_layer_copfeature = ((L_layer_imgfeature - mu_xL)/(std_xL + 1e-5)) * L_layer_txtfeature + L_layer_imgfeature
    M_layer_copfeature = ((M_layer_imgfeature - mu_xM)/(std_xM + 1e-5)) * M_layer_txtfeature + M_layer_imgfeature
    H_layer_copfeature = ((H_layer_imgfeature - mu_xH)/(std_xH + 1e-5)) * H_layer_txtfeature + H_layer_imgfeature


    L_layer_copfeature = self.img_model.avgpool(L_layer_copfeature)
    M_layer_copfeature = self.img_model.avgpool(M_layer_copfeature)
    H_layer_copfeature = self.img_model.avgpool(H_layer_copfeature)

    ALL_layer_copfeature = torch.cat((L_layer_copfeature, M_layer_copfeature, H_layer_copfeature), dim=1)
    ALL_layer_copfeature = ALL_layer_copfeature.view(ALL_layer_copfeature.size(0), -1)
    ALL_layer_copfeature = self.img_model.fc1(ALL_layer_copfeature)
    L_layer_copfeature = L_layer_copfeature.view(L_layer_copfeature.size(0), -1)
    M_layer_copfeature = M_layer_copfeature.view(M_layer_copfeature.size(0), -1)
    H_layer_copfeature = H_layer_copfeature.view(H_layer_copfeature.size(0), -1)

    representations = {"repres": ALL_layer_copfeature,
                       "Llayer_repres": L_layer_copfeature,
                       "Mlayer_repres": M_layer_copfeature,
                       "Hlayer_repres": H_layer_copfeature
                       }
    return representations    

class TIRG(ImgEncoderTextEncoderBase):
    """The TIRG model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        merged_dim = image_embed_dim + text_embed_dim

        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim)
        )

        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]

        dct_with_representations = {"repres": f}
        return dct_with_representations
      
class TIRGLastConv(ImgEncoderTextEncoderBase):
  """The TIGR model with spatial modification over the last conv layer.

  The method is described in
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
  "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
  CVPR 2019. arXiv:1812.07119
  """

  def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
    super(TIRGLastConv, self).__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)

    self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
    self.use_bert = use_bert
    merged_dim = image_embed_dim + text_embed_dim
    self.mod2d = torch.nn.Sequential(
        torch.nn.BatchNorm2d(merged_dim),
        torch.nn.Conv2d(merged_dim, merged_dim, [3, 3], padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(merged_dim, image_embed_dim, [3, 3], padding=1),
    )

    self.mod2d_gate = torch.nn.Sequential(
        torch.nn.BatchNorm2d(merged_dim),
        torch.nn.Conv2d(merged_dim, merged_dim, [3, 3], padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(merged_dim, image_embed_dim, [3, 3], padding=1),
    )

  def compose_img_text(self, imgs, text_query):
    #text_features = self.extract_text_feature(text_query)
    text_features = self.extract_text_feature(text_query, self.use_bert)

    x = imgs
    x = self.img_model.conv1(x)
    x = self.img_model.bn1(x)
    x = self.img_model.relu(x)
    x = self.img_model.maxpool(x)
    #print('L_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))

    x = self.img_model.layer1(x)
    x = self.img_model.layer2(x)
    #print('M_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))
    x = self.img_model.layer3(x)
    x = self.img_model.layer4(x)
    #print('H_layer shape:	{}x{}x{}'.format(x.shape[1],x.shape[2],x.shape[3]))

    # mod
    y = text_features
    y = y.reshape((y.shape[0], y.shape[1], 1, 1)).repeat(
        1, 1, x.shape[2], x.shape[3])
    z = torch.cat((x, y), dim=1)
    #print ('z shape:',z.shape)#z shape: torch.Size([200, 1024, 4, 6])
    t = self.mod2d(z)
    #print ('t shape:',t.shape)#t shape: torch.Size([200, 512, 4, 6])
    tgate = self.mod2d_gate(z)
    #print ('tgate shape:',tgate.shape)#tgate shape: torch.Size([200, 512, 4, 6])
    x = self.a[0] * F.sigmoid(tgate) * x + self.a[1] * t
    #print ('x shape:',x.shape)#x shape: torch.Size([200, 512, 4, 6])

    x = self.img_model.avgpool(x)
    #print ('x shape:',x.shape)#x shape: torch.Size([200, 512, 1, 1])
    x = x.view(x.size(0), -1)
    #print ('x shape:',x.shape)#x shape: torch.Size([200, 512])
    x = self.img_model.fc(x)
    #print ('x shape:',x.shape)#x shape: torch.Size([200, 512])
    
    dct_with_representations = {"repres": x}
    #print ('x:',x)
    return dct_with_representations
        
class ComplexProjectionModule(torch.nn.Module):

    def __init__(self, image_embed_dim =512, text_embed_dim = 768):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score

class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim =512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear
        
class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim =512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(16)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1024))
        theta_conv = self.mapping(final_vec)
        return theta_conv

class ComposeAE(ImgEncoderTextEncoderBase):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(),
            ConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features, CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False)):
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations

class RealConCatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        concat_x = torch.cat(x, -1)
        return concat_x

class RealLinearMapping(torch.nn.Module):
    """
    This is linear mapping from real space to image space.
    """

    def __init__(self, image_embed_dim =512, text_embed_dim= 768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        theta_linear = self.mapping(x)
        return theta_linear

class RealConvMapping(torch.nn.Module):
    """
    This is convoultional mapping from Real space to image space.
    """

    def __init__(self, image_embed_dim =512, text_embed_dim= 768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(20)

    def forward(self, x):
        concat_x = self.conv1(x.unsqueeze(1))
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1280))
        theta_conv = self.mapping(final_vec)
        return theta_conv

class RealSpaceConcatAE(ImgEncoderTextEncoderBase):
    """The RealSpaceConcatAE model.

    The method  in ablation study Table 5 (Concat in real space)
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            RealConCatModule(),
            RealLinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            RealConCatModule(),
            RealConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        theta_linear = self.encoderLinear((img_features, text_features))
        theta_conv = self.encoderWithConv((img_features, text_features))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations