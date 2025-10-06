import datetime
import json
import os
from dotenv import load_dotenv
from dateutil import parser
from tqdm import tqdm

from openai import OpenAI

load_dotenv('env.txt')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def process_reviews(path = './res/review.json'):
    with open(path,'r',encoding='utf=8') as f:
        review_lists = json.load(f)

    review_good, review_bad = [],[]

    current_date = datetime.datetime.now()
    date_boundary = current_date - datetime.timedelta(days = 180)

    for r in review_lists:
        review_date_str = r['date']

        try:
            review_date = parser.parse(review_date_str)
        except (ValueError, TypeError):
            review_date = current_date

        if review_date < date_boundary:
            continue

        if r['stars'] == 5:
            review_good.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')
        else:
            review_bad.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')

        reviews_good_text = '\n'.join(review_good)
        reviews_bad_text = '\n'.join(review_bad)

        return reviews_good_text, reviews_bad_text


def pairwise_eval(reviews, answer_a, answer_b):
    eval_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the Korean summaries provided by two
AI assistants to the set of user reviews on accommodations displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
[User Reviews]
{reviews}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""
    
    completion = client.chat.completions.create(
        model='gpt-4o-2024-05-13',
        messages=[{'role': 'user', 'content': eval_prompt}],
        temperature=0.0
    )

    return completion

PROMPT_BASELINE = f"""아래 숙소 리뷰에 대해 5문장 내로 요약해줘:"""

reviews, _ = process_reviews(path='./res/review.json')

def summarize(reviews, prompt, temperature=0.0, model='gpt-3.5-turbo-0125'):
    prompt = prompt + '\n\n' + reviews

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )

    return completion

print(summarize(reviews, PROMPT_BASELINE).choices[0].message.content)

summary_real_20240526 = '위치가 매우 우수한 숙박시설로, 인사동과 조계사, 경복궁 등 관광지에 도보로 이동할 수 있는 편리한 위치에 있습니다. 객실은 깔끔하며 직원들의 친절한 서비스와 청결한 시설이 인상적입니다. 주변에는 맛집과 편의시설이 많아 편리하며, 교통 접근성도 좋습니다. 전체적으로 만족도가 높고 자주 방문하고 싶은 곳으로 손꼽히는 숙소로 평가됩니다.'

print(pairwise_eval(reviews, summarize(reviews, PROMPT_BASELINE).choices[0].message.content, summary_real_20240526).choices[0].message.content)

eval_count = 10

summaries_baseline = [summarize(reviews, PROMPT_BASELINE, temperature=1.0).choices[0].message.content for _ in range(eval_count)]
summaries_baseline

def pairwise_eval_batch(reviews, answers_a, answers_b):
    a_cnt, b_cnt, draw_cnt = 0, 0, 0
    for i in tqdm(range(len(answers_a))):
        completion = pairwise_eval(reviews, answers_a[i], answers_b[i])
        verdict_text = completion.choices[0].message.content

        if '[[A]]' in verdict_text:
            a_cnt += 1
        elif '[[B]]' in verdict_text:
            b_cnt += 1
        elif '[[C]]' in verdict_text:
            draw_cnt += 1
        else:
            print('Evaluation Error')

    return a_cnt, b_cnt, draw_cnt

#wins, losses, ties = pairwise_eval_batch(reviews, summaries_baseline, [summary_real_20240526 for _ in range(len(summaries_baseline))])
#print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')

prompt = f"""당신은 요약 전문가입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

요약 결과는 다음 조건들을 충족해야 합니다:
1. 모든 문장은 항상 존댓말로 끝나야 합니다.
2. 숙소에 대해 소개하는 톤앤매너로 작성해주세요.
  2-1. 좋은 예시
    a) 전반적으로 좋은 숙소였고 방음도 괜찮았다는 평입니다.
    b) 재방문 예정이라는 평들이 존재합니다.
  2-2. 나쁜 예시
    a) 좋은 숙소였고 방음도 괜찮았습니다.
    b) 재방문 예정입니다.
3. 요약 결과는 최소 2문장, 최대 5문장 사이로 작성해주세요.
    
아래 숙소 리뷰들에 대해 요약해주세요:"""

eval_count = 10
summaries = [summarize(reviews, prompt, temperature=1.0).choices[0].message.content for _ in range(eval_count)]
wins, losses, ties = pairwise_eval_batch(reviews, summaries, [summary_real_20240526 for _ in range(len(summaries))])
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')