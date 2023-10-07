from fastapi import FastAPI
from pydantic import BaseModel
from generation import inference

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
    type: str
    personalise: Personalisation
    controller: str
    brand_name: str
    action: str
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
    template = "Create 1 subject line for an email campaign in a {tone} tone about {prompt} {str}. {personalize}.".format(            
        prompt=prompt,str=str,tone=tone, personalise=personalise)
    return template

@app.post("/generatesubject")
async def generatesubject(payload: InputPayload):
    message = payload.params
    personalise = personaliseForSubjectLine(message.personalise)
    about_brand = message.about_brand if message.about_brand else ""
    prompt = get_subject_prompt(message.prompt, about_brand, personalise, tone_map[message.tone])

    return inference(prompt)