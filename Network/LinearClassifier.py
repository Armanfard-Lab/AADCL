class LinCLS(nn.Module):
    def __init__(self, input_dim=512, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
