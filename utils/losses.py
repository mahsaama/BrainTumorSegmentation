import tensorflow as tf

# class FocalLoss(nn.modules.loss._WeightedLoss):

#     def __init__(self, gamma=0, size_average=None, ignore_index=-100,
#                  reduce=None, balance_param=1.0):
#         super(FocalLoss, self).__init__(size_average)
#         self.gamma = gamma
#         self.size_average = size_average
#         self.ignore_index = ignore_index
#         self.balance_param = balance_param

#     def forward(self, input, target):
        
#         # inputs and targets are assumed to be BatchxClasses
#         assert len(input.shape) == len(target.shape)
#         assert input.size(0) == target.size(0)
#         assert input.size(1) == target.size(1)
           
#         # compute the negative likelyhood
#         logpt = - F.binary_cross_entropy_with_logits(input, target)
#         pt = torch.exp(logpt)

#         # compute the loss
#         focal_loss = -( (1-pt)**self.gamma ) * logpt
#         balanced_focal_loss = self.balance_param * focal_loss
#         return balanced_focal_loss

def diceLoss(y_true, y_pred, class_weights):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

    num = tf.math.reduce_sum(tf.math.multiply(class_weights, tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3])))
    den = tf.math.reduce_sum(tf.math.multiply(class_weights, tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0,1,2,3])))+1e-5

    return 1-2*num/den

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2))
    fake_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2))

    disc_loss = 0.5*(real_loss + fake_loss)

    return disc_loss


def generator_loss(target, gen_output, disc_fake_output, class_weights, alpha):
    
    # generalized dice loss
    dice_loss = diceLoss(target, gen_output, class_weights)
    
    # disc loss
    disc_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2))
       
    # total loss
    gen_loss = alpha*dice_loss + disc_loss

    dice_percent = (1-dice_loss) * 100
    return gen_loss, dice_loss, disc_loss, dice_percent
