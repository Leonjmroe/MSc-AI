class BinarySearchTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def add(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTreeNode(value)
            else:
                self.left.add(value)
        else:
            if self.right is None:
                self.right = BinarySearchTreeNode(value)
            else:
                self.right.add(value)
