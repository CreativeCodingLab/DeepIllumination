import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 4, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(n_filters * 4 * 2, n_filters * 2, 4, 2, 1)
        self.deconv7 = nn.ConvTranspose2d(n_filters * 2 * 2, n_filters, 4, 2, 1)
        self.deconv8 = nn.ConvTranspose2d(n_filters * 2, n_channel_output, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(n_filters)
        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.batch_norm8(self.conv5(self.leaky_relu(encoder4)))
        encoder6 = self.batch_norm8(self.conv6(self.leaky_relu(encoder5)))
        encoder7 = self.batch_norm8(self.conv7(self.leaky_relu(encoder6)))
        encoder8 = self.conv8(self.leaky_relu(encoder7))

        decoder1 = self.dropout(self.batch_norm8(self.deconv1(self.relu(encoder8))))
        decoder1 = torch.cat((decoder1, encoder7), 1)
        decoder2 = self.dropout(self.batch_norm8(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2, encoder6), 1)
        decoder3 = self.dropout(self.batch_norm8(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3, encoder5), 1)
        decoder4 = self.batch_norm8(self.deconv4(self.relu(decoder3)))
        decoder4 = torch.cat((decoder4, encoder4), 1)
        decoder5 = self.batch_norm4(self.deconv5(self.relu(decoder4)))
        decoder5 = torch.cat((decoder5, encoder3), 1)
        decoder6 = self.batch_norm2(self.deconv6(self.relu(decoder5)))
        decoder6 = torch.cat((decoder6, encoder2),1)
        decoder7 = self.batch_norm(self.deconv7(self.relu(decoder6)))
        decoder7 = torch.cat((decoder7, encoder1), 1)
        decoder8 = self.deconv8(self.relu(decoder7))
        output = self.tanh(decoder8)
        return output

class D(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input + n_channel_output, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.conv5(self.leaky_relu(encoder4))
        output =  self.sigmoid(encoder5)
        return output
