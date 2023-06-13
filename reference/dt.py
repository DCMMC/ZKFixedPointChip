# [ref] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
from itertools import chain

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for row in dataset:
		for index in range(len(dataset[0])-1):
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	max_depth = max_depth - 1
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

def layers(root):
    queue = []
    result = []
    cnt = 0
    # If the root is not null, add it to the queue
    if root:
        queue.append(root)
    # While the queue is not empty, repeat steps 4-7
    while queue:
        # Get the size of the queue and initialize an empty list to store the nodes of the current level
        size = len(queue)
        level = []
        # Loop through the elements of the current level and add them to the list
        for i in range(size):
            node = queue.pop(0)
            print(node)
            if isinstance(node, int):
                print("yes")
                node = {'left': None, 'right': None, 'cls': node, 'index': cnt}
            else:
                node['index'] = cnt
            cnt += 1
            level.append(node)
            # For each element of the current level, add its children to the queue
            if node['left']:
                queue.append(node['left'])
            if node['right']:
                queue.append(node['right'])


        # Add the current level list to the result list
        result.append(level)
    return result

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# dataset = [[2.771244718,1.784783929,0],
# 	[1.728571309,1.169761413,0],
# 	[3.678319846,2.81281357,0],
# 	[3.961043357,2.61995032,0],
# 	[2.999208922,2.209014212,0],
# 	[7.497545867,3.162953546,1],
# 	[9.00220326,3.339047188,1],
# 	[7.444542326,0.476683375,1],
# 	[10.12493903,3.234550982,1],
# 	[6.642287351,3.319983761,1]]
dataset = [[2.771244718,1.784783929,1],
        [1.,1.,0],
        [3.,2.,0],
        [3.,2.,1],
        [2.,2.,0],
        [7.,3.,1],
        [9.00220326,3.339047188,1],
        [7.444542326,0.476683375,0],
        [10.12493903,3.234550982,1],
        [6.642287351,3.319983761,1]]
tree = build_tree(dataset, 4, 1)
print_tree(tree)
# print([i for i in chain.from_iterable(layers(tree))])
# print([[i['index'], i['value'], i['left']['index'], i['right']['index'], -1] if 'value' in i else [i['index'], 0, i['index'], i['index'], i['cls']] for i in chain.from_iterable(layers(tree))])
for row in dataset:
    prediction = predict(tree, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
