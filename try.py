def extract_values(lst):
    result = []

    def helper(sublist, index):
        if index < len(sublist):
            value = sublist[index]
            if isinstance(value, list):
                result.append(helper(lst[index - value], value[0]))
            else:
                result.append(value)
            return helper(sublist, index + 1)
        else:
            return result

    return helper(lst, 1)


# 测试例子
print(extract_values([[0, 1, 1, -1], [2, 3]]))  # 输出: [1, -1]
print(extract_values([[0, 1, 1, -1], [2, 3], [1]]))  # 输出: [-1]
