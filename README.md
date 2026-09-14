傳統的貝氏神經場（Bayesian Neural Fields, BayesNF）在處理連續型時空資料時，通常預設觀測模型為標準常態分配或卜瓦松分配。然而，在處理特定領域的時空動態分析（例如：分析台灣各鄉鎮區域的特定死亡率）時，資料往往會因為區域特性或極端值而呈現嚴重的「點膨脹」（Point-inflation）現象。本擴充套件透過掛載全新的 INFLATED_NORMAL 觀測模型，賦予 BayesNF 處理這類極端分佈的能力。模型會自動學習每個時空節點 $(s, t)$ 的膨脹機率與常態均值，進而大幅提升模型在複雜時空特徵下的預測能力。
無侵入式設計（Non-invasive Patching）： 透過 enable_mixture_normal() 函數即可在 Runtime 動態攔截並替換 BayesNF 內部的 likelihood model，完全不需要修改 BayesNF 的原始碼。  
雙頭神經網路架構（2-Head MLP Architecture）： 擴展了原有的 BayesianNeuralField1D 模型，建立共享主幹與兩個獨立的輸出分支，同時預測常態分佈的均值（$F$）以及點膨脹發生的 Logit 機率（$p$）。  
高度彈性的點質量近似（Point Mass Approximation）： 允許使用者自定義膨脹點的位置（透過 set_inflated_loc(v)），並使用極小變異（Scale = $10^{-3}$）的常態分佈來近似點質量，以維持優異的數值穩定性。  
