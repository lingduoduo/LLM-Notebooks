"""Build an evaluation dataset from Chinese criminal law, in English translation."""

import json
import logging
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalDatasetBuilder:
    """Build an evaluation dataset for Chinese criminal law Q&A (English)."""
    
    def __init__(self):
        self.simple_cases = []
        self.complex_cases = []
        
    def create_simple_cases(self) -> List[Dict[str, Any]]:
        """Create simple direct legal questions"""
        simple_cases = [
            {
                "id": "simple_1",
                "question": "What is the sentence for intentional homicide?",
                "expected_keywords": ["death penalty", "life imprisonment", "ten years or more"],
                "reference": "Criminal Law of the PRC, Article 232",
                "difficulty": "easy"
            },
            {
                "id": "simple_2",
                "question": "What is the threshold for filing a theft case?",
                "expected_keywords": ["1,000", "3,000", "relatively large amount"],
                "reference": "Criminal Law of the PRC, Article 264",
                "difficulty": "easy"
            },
            {
                "id": "simple_3", 
                "question": "How is drunk driving of a motor vehicle punished?",
                "expected_keywords": ["criminal detention", "fine", "licence"],
                "reference": "Criminal Law of the PRC, Article 133",
                "difficulty": "easy"
            },
            {
                "id": "simple_4",
                "question": "What are the sentencing tiers for fraud?",
                "expected_keywords": ["three years", "three to ten years", "ten years or more"],
                "reference": "Criminal Law of the PRC, Article 266",
                "difficulty": "easy"
            },
            {
                "id": "simple_5",
                "question": "What is the punishment for intentional injury causing grievous harm?",
                "expected_keywords": ["three to ten years", "fixed-term imprisonment"],
                "reference": "Criminal Law of the PRC, Article 234",
                "difficulty": "easy"
            },
            {
                "id": "simple_6",
                "question": "What are the aggravating circumstances for robbery?",
                "expected_keywords": ["home-invasion robbery", "repeated robbery", "huge amount"],
                "reference": "Criminal Law of the PRC, Article 263",
                "difficulty": "medium"
            },
            {
                "id": "simple_7",
                "question": "What are the elements of unlawful detention?",
                "expected_keywords": ["unlawful", "detention", "restriction of personal liberty"],
                "reference": "Criminal Law of the PRC, Article 238",
                "difficulty": "medium"
            },
            {
                "id": "simple_8",
                "question": "How are the monetary thresholds for embezzlement determined?",
                "expected_keywords": ["30,000 yuan", "200,000 yuan", "3,000,000 yuan"],
                "reference": "Criminal Law of the PRC, Article 383",
                "difficulty": "medium"
            },
            {
                "id": "simple_9",
                "question": "What is the threshold for filing a traffic-accident crime case?",
                "expected_keywords": ["death", "grievous injury", "property loss"],
                "reference": "Criminal Law of the PRC, Article 133",
                "difficulty": "easy"
            },
            {
                "id": "simple_10",
                "question": "How is picking quarrels and provoking trouble punished?",
                "expected_keywords": ["five years", "fixed-term imprisonment",
                                      "criminal detention", "public surveillance"],
                "reference": "Criminal Law of the PRC, Article 293",
                "difficulty": "easy"
            }
        ]
        
        return simple_cases
    
    def create_complex_cases(self) -> List[Dict[str, Any]]:
        """Create complex legal scenario questions"""
        complex_cases = [
            {
                "id": "complex_1",
                "question": """Following a financial dispute, Zhang broke into Li's home with a
                            knife intending to collect a debt. During the struggle Zhang stabbed
                            and grievously injured Li, and also took 50,000 yuan in cash from the
                            home. How should Zhang's conduct be characterized, and what criminal
                            penalties might he face?""",
                "expected_analysis": ["home invasion robbery", "intentional injury",
                                      "combined punishment for several crimes"],
                "reference": "Criminal Law, Articles 234 and 263",
                "difficulty": "hard",
                "requires_multi_query": True
            },
            {
                "id": "complex_2",
                "question": """Wang, the finance director of a state-owned enterprise, abused
                            his position to move 2,000,000 yuan of company funds into accounts he
                            controlled, using means such as falsely issued invoices. He then
                            invested the money in stocks and gained 500,000 yuan. After the case
                            surfaced he voluntarily returned all the proceeds. Analyze Wang's
                            legal liability.""",
                "expected_analysis": ["embezzlement", "misappropriation of public funds",
                                      "voluntary surrender", "return of proceeds"],
                "reference": "Criminal Law, Articles 382 and 384",
                "difficulty": "hard",
                "requires_multi_query": True
            },
            {
                "id": "complex_3",
                "question": """Zhao drove after drinking and sped through the city, striking
                            pedestrian Chen who was crossing the road and killing him instantly.
                            Zhao then fled the scene. The next day, urged by his family, he
                            surrendered to the police. Which offences is Zhao suspected of, and
                            what factors should sentencing take into account?""",
                "expected_analysis": ["traffic accident crime", "dangerous driving",
                                      "fleeing the scene", "voluntary surrender"],
                "reference": "Criminal Law, Article 133",
                "difficulty": "hard",
                "requires_multi_query": True
            },
            {
                "id": "complex_4",
                "question": """Liu posted false investment offers on an online platform,
                            promising guaranteed high returns, and defrauded 30 investors of
                            5,000,000 yuan in total. He spent 2,000,000 yuan on personal
                            extravagance and used 3,000,000 yuan to repay earlier debts. How
                            should Liu's conduct be characterized, and what sentence is likely?""",
                "expected_analysis": ["fraud", "especially huge amount", "multiple victims"],
                "reference": "Criminal Law, Article 266",
                "difficulty": "hard",
                "requires_multi_query": True
            },
            {
                "id": "complex_5",
                "question": """Sun and Qian conspired to steal from a shopping mall. Sun kept
                            watch while Qian went inside to carry out the theft. A security guard
                            discovered Qian mid-theft, and Qian struck the guard, causing minor
                            injury, in order to escape. The pair stole property worth 80,000 yuan.
                            Analyze the criminal liability of each.""",
                "expected_analysis": ["joint crime", "theft", "robbery", "transformation"],
                "reference": "Criminal Law, Articles 264 and 269",
                "difficulty": "hard",
                "requires_multi_query": True
            }
        ]
        
        return complex_cases
    
    def build_dataset(self, output_path: str = "legal_qa_dataset.json"):
        """Build and save the complete dataset"""
        dataset = {
            "simple_cases": self.create_simple_cases(),
            "complex_cases": self.create_complex_cases(),
            "metadata": {
                "total_cases": 15,
                "simple_count": 10,
                "complex_count": 5,
                "domain": "Chinese Criminal Law",
                "purpose": "Evaluate agentic vs non-agentic RAG performance"
            }
        }
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        return dataset


def create_legal_documents() -> List[Dict[str, str]]:
    """Create sample legal documents for the knowledge base.

    English translations of the cited PRC Criminal Law articles. Article numbers
    are preserved so the references in the evaluation cases still line up.
    """
    documents = [
        {
            "doc_id": "criminal_law_homicide",
            "title": "Criminal Law - Intentional Homicide",
            "content": """Article 232 [Intentional homicide] Whoever intentionally kills another
            shall be sentenced to death, life imprisonment or fixed-term imprisonment of ten years
            or more; where the circumstances are relatively minor, to fixed-term imprisonment of
            three to ten years.

            Intentional homicide is the intentional and unlawful deprivation of another person's
            life. The protected interest is the right to life. The legal basis is Article 232 of
            the Criminal Law of the PRC.

            Sentencing tiers:
            1. Serious circumstances: death penalty, life imprisonment, or ten years or more
            2. Relatively minor circumstances: three to ten years of fixed-term imprisonment

            Relatively minor circumstances typically include excessive self-defence, killing out
            of righteous indignation, and cases where the victim was at fault."""
        },
        {
            "doc_id": "criminal_law_theft",
            "title": "Criminal Law - Theft",
            "content": """Article 264 [Theft] Whoever steals public or private property in a
            relatively large amount, or commits repeated theft, home-invasion theft, theft while
            carrying a weapon, or pickpocketing, shall be sentenced to fixed-term imprisonment of
            not more than three years, criminal detention or public surveillance, and shall also
            or only be fined; where the amount is huge or other serious circumstances exist, to
            three to ten years and a fine; where the amount is especially huge or other especially
            serious circumstances exist, to ten years or more or life imprisonment, and a fine or
            confiscation of property.

            Thresholds for filing a theft case:
            1. Relatively large amount: generally 1,000 to 3,000 yuan or more
            2. Huge amount: generally 30,000 to 100,000 yuan or more
            3. Especially huge amount: generally 300,000 to 500,000 yuan or more

            Special cases: repeated theft (three or more times within two years), home-invasion
            theft, theft while carrying a weapon and pickpocketing constitute theft regardless of
            the amount involved."""
        },
        {
            "doc_id": "criminal_law_fraud",
            "title": "Criminal Law - Fraud",
            "content": """Article 266 [Fraud] Whoever defrauds public or private property in a
            relatively large amount shall be sentenced to fixed-term imprisonment of not more than
            three years, criminal detention or public surveillance, and shall also or only be
            fined; where the amount is huge or other serious circumstances exist, to three to ten
            years and a fine; where the amount is especially huge or other especially serious
            circumstances exist, to ten years or more or life imprisonment, and a fine or
            confiscation of property.

            Sentencing tiers for fraud:
            1. Relatively large amount (3,000 to 10,000 yuan or more): up to three years,
               criminal detention or public surveillance
            2. Huge amount (30,000 to 100,000 yuan or more): three to ten years
            3. Especially huge amount (500,000 yuan or more): ten years or more, or life

            Fraud is the act of obtaining public or private property in a relatively large amount
            by fabricating facts or concealing the truth, with intent to unlawfully possess it."""
        },
        {
            "doc_id": "criminal_law_robbery",
            "title": "Criminal Law - Robbery",
            "content": """Article 263 [Robbery] Whoever robs public or private property by
            violence, coercion or other means shall be sentenced to fixed-term imprisonment of
            three to ten years and a fine; in any of the following circumstances, to ten years or
            more, life imprisonment or death, and a fine or confiscation of property:

            (1) home-invasion robbery;
            (2) robbery on public transport;
            (3) robbery of a bank or other financial institution;
            (4) repeated robbery, or robbery of a huge amount;
            (5) robbery causing grievous injury or death;
            (6) robbery while impersonating military or police personnel;
            (7) armed robbery;
            (8) robbery of military supplies, or of emergency, disaster-relief or relief supplies.

            The aggravating circumstances for robbery are the eight cases above; where any applies,
            the minimum sentence is ten years of fixed-term imprisonment."""
        },
        {
            "doc_id": "criminal_law_injury",
            "title": "Criminal Law - Intentional Injury",
            "content": """Article 234 [Intentional injury] Whoever intentionally injures another
            shall be sentenced to fixed-term imprisonment of not more than three years, criminal
            detention or public surveillance. Whoever commits the offence in the preceding
            paragraph and causes grievous injury shall be sentenced to three to ten years; whoever
            causes death, or causes severe disability through grievous injury by especially cruel
            means, shall be sentenced to ten years or more, life imprisonment or death.

            Sentencing for intentional injury:
            1. Minor injury: up to three years, criminal detention or public surveillance
            2. Grievous injury: three to ten years of fixed-term imprisonment
            3. Death, or severe disability by especially cruel means: ten years or more, life
               imprisonment or death

            Grievous injury means: causing loss of a limb or disfigurement; causing loss of
            hearing, sight or the function of another organ; or otherwise seriously harming
            physical health."""
        },
        {
            "doc_id": "criminal_law_traffic",
            "title": "Criminal Law - Traffic Accident and Dangerous Driving",
            "content": """Article 133 [Traffic accident crime] Whoever violates traffic and
            transport regulations and thereby causes a major accident, resulting in grievous
            injury, death or major loss of public or private property, shall be sentenced to
            fixed-term imprisonment of not more than three years or criminal detention; whoever
            flees after the accident, or where other especially bad circumstances exist, to three
            to seven years; where fleeing causes a death, to seven years or more.

            Article 133(a) [Dangerous driving] Whoever drives a motor vehicle on a road in any of
            the following circumstances shall be sentenced to criminal detention and a fine:

            (1) racing in pursuit, where the circumstances are bad;
            (2) driving a motor vehicle while intoxicated;
            (3) operating a school bus or passenger transport service while seriously exceeding
                the rated passenger capacity, or seriously exceeding the speed limit;
            (4) transporting hazardous chemicals in violation of safety regulations, endangering
                public safety.

            Threshold for intoxicated driving: blood alcohol concentration of 80 mg/100 ml or
            above."""
        },
        {
            "doc_id": "criminal_law_corruption",
            "title": "Criminal Law - Embezzlement",
            "content": """Article 382 [Embezzlement] A state functionary who takes advantage of
            their position to misappropriate, steal, obtain by fraud or otherwise unlawfully
            possess public property commits embezzlement.

            Article 383 [Punishment for embezzlement] Whoever commits embezzlement shall be
            punished according to the seriousness of the circumstances:

            (1) where the amount is relatively large or other relatively serious circumstances
                exist: up to three years of fixed-term imprisonment or criminal detention, and a
                fine.
            (2) where the amount is huge or other serious circumstances exist: three to ten years,
                and a fine or confiscation of property.
            (3) where the amount is especially huge or other especially serious circumstances
                exist: ten years or more or life imprisonment, and a fine or confiscation of
                property; where the amount is especially huge and the interests of the state and
                the people suffer especially heavy losses, life imprisonment or death, and
                confiscation of property.

            Monetary thresholds for embezzlement:
            1. Relatively large: 30,000 yuan up to 200,000 yuan
            2. Huge: 200,000 yuan up to 3,000,000 yuan
            3. Especially huge: 3,000,000 yuan or more"""
        }
    ]
    
    return documents


if __name__ == "__main__":
    # Build evaluation dataset
    builder = LegalDatasetBuilder()
    dataset = builder.build_dataset("legal_qa_dataset.json")
    
    print(f"Dataset created with {len(dataset['simple_cases'])} simple cases and {len(dataset['complex_cases'])} complex cases")
    
    # Create legal documents
    documents = create_legal_documents()
    
    # Save documents
    with open("legal_documents.json", 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(documents)} legal documents for knowledge base")
