from core.hypothesis_testing import anova_oneway, t_test

if __name__ == '__main__':
    data = [
        [77, 88, 77, 85, 81, 72, 80, 80, 76, 84],
        [74, 88, 77, 93, 91, 95, 85, 88, 93, 79],
        [93, 94, 95, 83, 94, 94, 85, 91, 90, 96]
    ]
    print(anova_oneway(data))

    data1 = [77, 88, 77, 85, 81, 72, 80, 80, 76, 84]
    data2 = [74, 88, 77, 93, 91, 95, 85, 88, 93, 79]
    data = [data1, data2]
    print(t_test(data1, data2, tail="both", equal=True, mu=0))
    print(anova_oneway(data))
    print(t_test(data1, data2, tail="both", equal=True, mu=0)[0] ** 2)
