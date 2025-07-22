from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load your knowledge base
knowledge_base = [
    {"question": "តើអ្នកជានរណា?", "answer": "ខ្ញុំជា SmartBot ដែលត្រូវបានបង្កើតឡើងដើម្បីឆ្លើយសំណួររទាក់មេរៀន AI for robotics។"},
    {"question": "Who are you?", "answer": "I am SmartBot, an AI designed to answer your questions."},
    {"question": "តើអ្នកអាចធ្វើអ្វីខ្លះ?", "answer": "ខ្ញុំអាចឆ្លើយសំណួរទាក់់ទងនឹងក្រុមខ្ងុំធ្វើ Slide: AI/ML"},
    {"question": "What can you do?", "answer": "I can answer questions related to my team Slide: AI/ML."},
    {"question": "តើអ្វីទៅដែលហៅ AI?", "answer": "គឺជាការបង្កើតប្រព័ន្ធឬម៉ាស៊ីនដែលអាចបង្ហាញនូវសមត្ថភាពស្រដែងនិងមនុស្សដូចជា៖ការរៀន,គិតវែកញែក ,ដោះស្រាយបញ្ហា និងឆ្លើយជាមួយនឹងមនុស្សបាន "},
    {"question": "What is AI?", "answer": "AI is the creation of systems or machines that can demonstrate human-like abilities such as learning, reasoning, problem-solving, and interacting with people."},
    {"question": "តើ AI មានប៉ុន្មានប្រភេទ?", "answer": "AI មានបីប្រភេទគឺ Narrow AI, General AI, និ Supper AI ។ សូមមើលសំអិតក្នុង Slide ក្រុមរបស់ខ្ងុំ"},
    {"question": "How many types of AI are there?", "answer": "There are three types of AI: Narrow AI, Generator AI, and Super AI. See the details in my group slide."},
    {"question": "តើ Key នៃ AI មានអ្វីខ្លះ?", "answer": "Machine Learning, Deep learning , Netural Language Processing, Computer Vision, Robotics, Expert System, Reinforcement Learning"},
    {"question": "What are the keys to AI?", "answer": "Machine Learning, Deep learning , Netural Language Processing, Computer Vision, Robotics, Expert System, Reinforcement Learning"},
    {"question": "តើ AI ដំណើរការយ៉ាងដូចម្តេច?", "answer": "1. Data Collection :AI ត្រូវការទិន្នន័យដើម្បីរៀន។ ទិន្នន័យអាចមកពី៖អ្នកប្រើប្រាស់:\n1.Sensor,Database, Web:\n2. Data Preprocessing : ទិន្នន័យត្រូវបានសម្អាត និងរៀបចំឲ្យសមរម្យសម្រាប់ការបណ្តុះម៉ូដែល៖ការលុប Missing Values,ការបម្លែងទិន្នន័យ, ការបំបាត់ Noise"},
    {"question": "How does AI work", "answer": "You're welcome! Is there anything else I can assist you with?"},
    {"question": "AI for robotic", "answer": "•   ឲ្យ Robot មានសមត្ថភាព “គិត” និង “រៀន” ដូចមនុស្ស។ បង្កើតប្រព័ន្ធដែលអាចដោះស្រាយបញ្ហា និងធ្វើសកម្មភាពដោយខ្លួនឯងដោយប្រើ:\n 1.machine learning \n 2.computer vision \n 3.reinforceming learning \n 4.NLP\n 5. Sensor fusion."},
    {"question": "Hello", "answer": "Hello! How can I help you today?"},
    {"question": "តើខ្ញុំអាចចូលរៀនវគ្គសិក្សានេះដោយរបៀបណា?", "answer": "អ្នកអាចចុះឈ្មោះចូលរៀនវគ្គសិក្សានេះតាមរយៈគេហទំព័ររបស់យើងនៅផ្នែក 'វគ្គសិក្សា'។"},
    {"question": "How can I enroll in this course?", "answer": "You can enroll in this course through our website in the 'Courses' section."},
    {"question": "តើខ្ញុំអាចរកឯកសារសិក្សានៅឯណា?", "answer": "ឯកសារសិក្សាទាំងអស់មាននៅក្នុងប្រព័ន្ធគ្រប់គ្រងការសិក្សា (LMS) របស់អ្នក។"},
    {"question": "Where can I find study materials?", "answer": "All study materials are available in your Learning Management System (LMS)."},
]


kb_df = pd.DataFrame(knowledge_base)
questions = kb_df['question'].tolist()
answers = kb_df['answer'].tolist()

# Load the model and compute embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Response logic
def get_chatbot_response(user_query, question_embeddings, questions, answers, model, similarity_threshold=0.65):
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_query_embedding, question_embeddings)[0]
    max_similarity_score = torch.max(cosine_scores).item()
    best_index = torch.argmax(cosine_scores).item()

    if max_similarity_score >= similarity_threshold:
        return answers[best_index]
    else:
        return "ខ្ញុំមិនយល់សំណួររបស់អ្នកទេ។ សូមសួរផ្សេងទៀត។"

# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("សួស្តី! ខ្ញុំជាបុត SmartBot 🤖។ សូមសួរអ្វីក៏បាន។")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    response = get_chatbot_response(user_input, question_embeddings, questions, answers, model)
    await update.message.reply_text(response)

# Run the bot
def main():
    app = ApplicationBuilder().token("7797460165:AAEN6bILORAdCJ12poxH9cuK5oupi0ztQIM").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
