# 商品图片识别与检索系统

## 功能

- 用户上传商品图片
- 使用 OpenAI gpt-4o-mini（视觉输入）解析图片内容为文字描述
- 使用LlamaIndex在商品数据库中检索相关商品

## 安装

1. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

## 系统流程

1. 用户通过Streamlit界面上传商品图片
2. 系统调用 OpenAI gpt-4o-mini 分析图片内容，生成详细的文字描述
3. 将文字描述传递给LlamaIndex，在预先建立的商品向量数据库中检索相关商品
4. 展示检索结果给用户

## 注意事项

- 需要有效的 OpenAI API 密钥（OPENAI_API_KEY）才能使用 gpt-4o-mini
- 示例中使用了简单的内存数据库，实际应用中可以替换为真实的商品数据库