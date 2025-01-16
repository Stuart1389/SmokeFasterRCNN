"""
def regression_distil(device, bbox_smooth_loss, student_bbox_delta, teacher_bbox_delta,
                      student_targets, teacher_targets, margin=0.01, penalty=0.01):
    #penalty: higher = learn from teacher, lower = learning on its own


    student_bbox_diff = student_bbox_delta - student_targets
    teacher_bbox_diff = teacher_bbox_delta - teacher_targets
    # square vals
    student_dif_squ = student_bbox_diff.pow(2)
    teacher_dif_squ = teacher_bbox_diff.pow(2)
    # iterate
    dim = range(1, len(student_bbox_diff.shape))
    for i in sorted(dim, reverse=True):
        student_sum = student_dif_squ.sum(i)
        teacher_sum = teacher_dif_squ.sum(i)
    zero_tens = torch.zeros(student_sum.shape).to(device)
    # comparing teacher results to ground truth to determine if we want to use teacher to help
    hint_suc = torch.where((student_sum + margin <= teacher_sum), zero_tens, student_sum)
    regression_loss = bbox_smooth_loss + penalty * hint_suc.mean()

    return regression_loss, regression_loss.mean(), student_sum.mean(), teacher_sum.mean()
"""

"""
# rpn regression
student_logit_rpn_box_reg = student_logits["logit_rpn_box_reg"]
teacher_logit_rpn_box_reg = teacher_logits["logit_rpn_box_reg"]

rpn_reg, rpn_reg_soft, _, _ = regression_distil(device, loss_rpn_box_reg, student_logit_rpn_box_reg, teacher_logit_rpn_box_reg,
                             student_rpn_gt_targets, teacher_rpn_gt_targets)

#roi regression
student_logit_bbox_delta = student_logits["logit_bbox_delta"]
teacher_logit_bbox_delta = teacher_logits["logit_bbox_delta"]

roi_reg, roi_reg_soft, _, _ = regression_distil(device, loss_box_reg, student_logit_bbox_delta, teacher_logit_bbox_delta,
                             student_roi_gt_targets, teacher_roi_gt_targets)

distil_loss_dict = {
    'obj_com': obj_com,
    'cls_com': cls_com,
    'rpn_reg': rpn_reg,
    'roi_reg': roi_reg
}
for key, value in distil_loss_dict.items():
    print(f'{key}: {value}')

losses_reduced = sum(loss for loss in distil_loss_dict.values())
loss_value = losses_reduced.item()
print(loss_value)
"""


def get_distil_loss(device, student_logits, teacher_logits, loss_classifier, loss_box_reg,
                    loss_objectness, loss_rpn_box_reg, teacher_roi_gt_targets, teacher_rpn_gt_targets,
                    student_roi_gt_targets, student_rpn_gt_targets, teacher_features, adapted_student_features, epoch,
                    max_epoch):
    # print(student_logits.keys())
    # Extract logits
    # rpn classification
    student_logit_objectness = student_logits["logit_objectness"]
    teacher_logit_objectness = teacher_logits["logit_objectness"]
    obj_com, obj_soft = classification_distil(student_logit_objectness, teacher_logit_objectness, 0.1, loss_objectness)
    # print(obj_com, obj_soft)

    # roi classification
    student_logit_classification = student_logits["logit_classification"]
    teacher_logit_classification = teacher_logits["logit_classification"]
    cls_com, cls_soft = classification_distil(student_logit_classification, teacher_logit_classification, 0.1,
                                              loss_classifier)

    # rpn regression
    student_logit_rpn_box_reg = student_logits["logit_rpn_box_reg"]
    teacher_logit_rpn_box_reg = teacher_logits["logit_rpn_box_reg"]

    rpn_reg, rpn_reg_soft, _, _ = regression_distil(device, loss_rpn_box_reg, student_logit_rpn_box_reg,
                                                    teacher_logit_rpn_box_reg,
                                                    student_rpn_gt_targets, teacher_rpn_gt_targets)

    # roi regression
    student_logit_bbox_delta = student_logits["logit_bbox_delta"]
    teacher_logit_bbox_delta = teacher_logits["logit_bbox_delta"]

    roi_reg, roi_reg_soft, _, _ = regression_distil(device, loss_box_reg, student_logit_bbox_delta,
                                                    teacher_logit_bbox_delta,
                                                    student_roi_gt_targets, teacher_roi_gt_targets)

    distil_loss_dict = {
        'obj_com': obj_com,
        'cls_com': cls_com,
        'rpn_reg': rpn_reg,
        'roi_reg': roi_reg
    }
    # for key, value in distil_loss_dict.items():
    # print(f'{key}: {value}')

    losses_reduced = sum(loss for loss in distil_loss_dict.values())
    loss_value = losses_reduced.item()
    # print(loss_value)

    return obj_com, cls_com


def regression_distil(device, bbox_smooth_loss, student_bbox_delta, teacher_bbox_delta,
                      student_targets, teacher_targets, margin=0.01, penalty=0.01):
    # penalty: higher = learn from teacher, lower = learning on its own

    student_bbox_diff = student_bbox_delta - student_targets
    teacher_bbox_diff = teacher_bbox_delta - teacher_targets
    # square vals
    student_dif_squ = student_bbox_diff.pow(2)
    teacher_dif_squ = teacher_bbox_diff.pow(2)
    # iterate
    dim = range(1, len(student_bbox_diff.shape))
    for i in sorted(dim, reverse=True):
        student_sum = student_dif_squ.sum(i)
        teacher_sum = teacher_dif_squ.sum(i)
    zero_tens = torch.zeros(student_sum.shape).to(device)
    hint_suc = torch.where((student_sum + margin <= teacher_sum), zero_tens, student_sum)
    regression_loss = bbox_smooth_loss + penalty * hint_suc.mean()

    return regression_loss, regression_loss.mean(), student_sum.mean(), teacher_sum.mean()


# KNOWLEGE DISTIL
object_loss, class_loss = get_distil_loss(device, student_logits, teacher_logits,
                                          loss_dict_reduced['loss_classifier'].item(),
                                          loss_dict_reduced['loss_box_reg'].item(),
                                          loss_dict_reduced['loss_objectness'].item(),
                                          loss_dict_reduced['loss_rpn_box_reg'].item(), teacher_roi_gt_targets,
                                          teacher_rpn_gt_targets,
                                          student_roi_gt_targets, student_rpn_gt_targets, teacher_features,
                                          adapted_student_features, epoch, max_epoch)

def classification_distil(student_logits, teacher_logits, weight_factor, student_loss, temperature=1):
    #print("student :", student_logits)
    #print("teacher :", teacher_logits)

    # Ensure logits are floats
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

    # Compute softmax probabilities with temperature scaling
    # binary classification
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    #student_probs = torch.softmax(student_logits / temperature, dim=1)
    #teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)

    # Calculate soft loss (knowledge distillation loss)
    soft_prob = torch.sum(teacher_probs * torch.log(student_probs), dim=1)  # era P_s è e-10
    wc = torch.ones(student_logits.shape[0]).cuda().float()
    soft_loss = -torch.mean(soft_prob*wc)


    # Combine the hard loss from the student with the soft loss
    combined_loss = weight_factor * student_loss + (1 - weight_factor) * soft_loss
    #print("combined_loss: ", combined_loss)
    #print("soft loss: ", soft_loss)
    return combined_loss, soft_loss