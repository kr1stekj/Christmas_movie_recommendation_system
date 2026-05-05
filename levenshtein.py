def levenshtein(word1, word2):
    n = len(word1)
    m = len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # удаление
                    dp[i][j - 1],  # вставка
                    dp[i - 1][j - 1]  # замена
                )
            # Дополнительная операция транспозиции
            if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[n][m]

if __name__ == '__main__':
    result = levenshtein(input("Слово 1"), input("Слово 2"))
    print(result)
