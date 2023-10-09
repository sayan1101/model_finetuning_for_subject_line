from fastapi import FastAPI
from pydantic import BaseModel
from generation import inference
import json
from typing import Optional

app = FastAPI()

tone_map = {
    "AWESTRUCK": "Enthusiastic",
    "BOLD": "Bold",
    "COMPASSIONATE": "Compassionate",
    "CONVINCING": "Convincing",
    "ENTHUSIASTIC": "Enthusiastic",
    "FORMAL": "Formal",
    "FRIENDLY": "Friendly",
    "JOYFUL": "Joyful",
    "LUXURY": "Luxury",
    "PROFESSIONAL": "Professional",
    "RELAXED": "Relaxed",
}

class Personalisation(BaseModel):
    isFirstName: bool
    isLastName: bool
    isShopName: bool
    isDiscountCode: bool
    isShortUrl: bool
    isShopUrl: bool
    isUnsubscribeUrl: bool

class Params(BaseModel):
    prompt: str
    tone: str
    type: Optional[str]
    personalise: str
    controller: Optional[str]
    action: Optional[str]
    about_brand: str

class InputPayload(BaseModel):
    params: Params

def personaliseForSubjectLine(options):
    template = ""
    firstNameReplacement = "Include {{contact.first_name|default:there}}"
    lastNameReplacement = "Include {{contact.last_name|default:there}}"
    shopNameReplacement = "Include {{shop_name}}"
    if options.isFirstName:
        template += firstNameReplacement
    if options.isLastName:
        template += lastNameReplacement
    if options.isShopName:
        template += shopNameReplacement
    return template

def get_subject_prompt(prompt, about_brand, personalise, tone):
    str = ""
    if about_brand != "":
        str = "for a company with about us '{about_brand}'".format(about_brand=about_brand)
    template = """###Instruction:
Create 1 subject line for an email campaign in a {tone} tone about {prompt} {str}. {personalise}.

### Response:
""".format(            
        prompt=prompt,str=str,tone=tone, personalise=personalise)
    return template

@app.post("/generatesubject")
async def generatesubject(payload: InputPayload):
    print(payload)
    message = payload.params
    print(message)
    print(message.personalise)
    personalise = Personalisation.parse_raw(message.personalise)
    print(personalise)
    personalised = personaliseForSubjectLine(personalise)
    about_brand = message.about_brand if message.about_brand else ""
    prompt = get_subject_prompt(message.prompt, about_brand, personalised, tone_map[message.tone])
    print(prompt)

    return inference(prompt)