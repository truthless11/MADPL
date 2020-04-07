import copy
import random
import torch
from datamanager import expand_da
from copy import deepcopy
from tracker import StateTracker
from goal_generator import GoalGenerator
from utils import init_goal, init_session

REF_USR_DA = {
    'Attraction': {
        'area': 'Area', 'type': 'Type', 'name': 'Name',
        'entrance fee': 'Fee', 'address': 'Addr',
        'postcode': 'Post', 'phone': 'Phone'
    },
    'Hospital': {
        'department': 'Department', 'address': 'Addr', 'postcode': 'Post',
        'phone': 'Phone'
    },
    'Hotel': {
        'type': 'Type', 'parking': 'Parking', 'pricerange': 'Price',
        'internet': 'Internet', 'area': 'Area', 'stars': 'Stars',
        'name': 'Name', 'stay': 'Stay', 'day': 'Day', 'people': 'People',
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Police': {
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Restaurant': {
        'food': 'Food', 'pricerange': 'Price', 'area': 'Area',
        'name': 'Name', 'time': 'Time', 'day': 'Day', 'people': 'People',
        'phone': 'Phone', 'postcode': 'Post', 'address': 'Addr'
    },
    'Taxi': {
        'leaveAt': 'Leave', 'destination': 'Dest', 'departure': 'Depart', 'arriveBy': 'Arrive',
        'car type': 'Car', 'phone': 'Phone'
    },
    'Train': {
        'destination': 'Dest', 'day': 'Day', 'arriveBy': 'Arrive',
        'departure': 'Depart', 'leaveAt': 'Leave', 'people': 'People',
        'duration': 'Time', 'price': 'Ticket', 'trainID': 'Id'
    }
}

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None, 'Open': None
    },
    'Hospital': {
        'Department': 'department', 'Addr': 'address', 'Post': 'postcode',
        'Phone': 'phone', 'none': None
    },
    'Booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'Hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Ref': "ref", 'Stars': "stars", 'Type': "type",
        'none': None
    },
    'Restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Ref': "ref",
        'none': None
    },
    'Taxi': {
        'Arrive': "arriveBy", 'Car': "car type", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "phone",
        'none': None
    },
    'Train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination", 'Id': "trainID",
        'Leave': "leaveAt", 'People': "people", 'Ref': "ref",
        'Time': "duration", 'none': None, 'Ticket': 'price',
    },
    'Police': {
        'Addr': "address", 'Post': "postcode", 'Phone': "phone"
    },
}

SELECTABLE_SLOTS = {
    'Attraction': ['area', 'entrance fee', 'name', 'type'],
    'Hospital': ['department'],
    'Hotel': ['area', 'internet', 'name', 'parking', 'pricerange', 'stars', 'type'],
    'Restaurant': ['area', 'name', 'food', 'pricerange'],
    'Taxi': [],
    'Train': [],
    'Police': [],
}

INFORMABLE_SLOTS = ["Fee", "Addr", "Area", "Stars", "Internet", "Department", "Choice", "Ref", "Food", "Type", "Price",\
                    "Stay", "Phone", "Post", "Day", "Name", "Car", "Leave", "Time", "Arrive", "Ticket", None, "Depart",\
                    "People", "Dest", "Parking", "Open", "Id"]

REQUESTABLE_SLOTS = ['Food', 'Area', 'Fee', 'Price', 'Type', 'Department', 'Internet', 'Parking', 'Stars', 'Type']

# Information required to finish booking, according to different domain.
booking_info = {'Train': ['People'],
                'Restaurant': ['Time', 'Day', 'People'],
                'Hotel': ['Stay', 'Day', 'People']}

# Alphabet used to generate phone number
digit = '0123456789'


class SystemRule(StateTracker):
    ''' Rule-based bot. Implemented for Multiwoz dataset. '''

    recommend_flag = -1
    choice = ""

    def __init__(self, data_dir, cfg):
        super(SystemRule, self).__init__(data_dir, cfg)
        self.last_state = {}
        self.goal_gen = GoalGenerator(data_dir, cfg,
                                      goal_model_path='processed_data/goal_model.pkl',
                                      corpus_path=cfg.data_file)

    def reset(self, random_seed=None):
        self.last_state = init_belief_state()
        self.time_step = 0
        self.topic = ''
        self.goal = self.goal_gen.get_user_goal(random_seed)
        
        dummy_state, dummy_goal = init_session(-1, self.cfg)
        init_goal(dummy_goal, dummy_state['goal_state'], self.goal, self.cfg)
        
        domain_ordering = self.goal['domain_ordering']
        dummy_state['next_available_domain'] = domain_ordering[0]
        dummy_state['invisible_domains'] = domain_ordering[1:]
        
        dummy_state['user_goal'] = dummy_goal
        self.evaluator.add_goal(dummy_goal)
        
        return dummy_state
        
    def _action_to_dict(self, das):
        da_dict = {}
        for da, value in das.items():
            domain, intent, slot = da.split('-')
            if domain != 'general':
                domain = domain.capitalize()
            if intent in ['inform', 'request']:
                intent = intent.capitalize()
            domint = '-'.join((domain, intent))
            if domint not in da_dict:
                da_dict[domint] = []
            da_dict[domint].append([slot.capitalize(), value])
        return da_dict
    
    def _dict_to_vec(self, das):
        da_vector = torch.zeros(self.cfg.a_dim, dtype=torch.int32)
        expand_da(das)
        for domint in das:
            pairs = das[domint]
            for slot, p, value in pairs:
                da = '-'.join((domint, slot, p)).lower()
                if da in self.cfg.da2idx:
                    idx = self.cfg.da2idx[da]
                    da_vector[idx] = 1
        return da_vector
    
    def step(self, s, usr_a):
        """
        interact with simulator for one user-sys turn
        """
        # update state with user_act
        current_s = self.update_belief_usr(s, usr_a)
        da_dict = self._action_to_dict(current_s['user_action'])
        state = self._update_state(da_dict)
        sys_a = self.predict(state)
        sys_a = self._dict_to_vec(sys_a)
        
        # update state with sys_act
        next_s = self.update_belief_sys(current_s, sys_a)
        return next_s

    def predict(self, state):
        """
        Args:
            State, please refer to util/state.py
        Output:
            DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """

        if self.recommend_flag != -1:
            self.recommend_flag += 1

        self.kb_result = {}

        DA = {}

        if 'user_action' in state and (len(state['user_action']) > 0):
            user_action = state['user_action']
        else:
            user_action = check_diff(self.last_state, state)

        # Debug info for check_diff function

        self.last_state = state

        for user_act in user_action:
            domain, intent_type = user_act.split('-')

            # Respond to general greetings
            if domain == 'general':
                self._update_greeting(user_act, state, DA)

            # Book taxi for user
            elif domain == 'Taxi':
                self._book_taxi(user_act, state, DA)

            elif domain == 'Booking':
                self._update_booking(user_act, state, DA)

            # User's talking about other domain
            elif domain != "Train":
                self._update_DA(user_act, user_action, state, DA)

            # Info about train
            else:
                self._update_train(user_act, user_action, state, DA)

            # Judge if user want to book
            self._judge_booking(user_act, user_action, DA)

            if 'Booking-Book' in DA:
                if random.random() < 0.5:
                    DA['general-reqmore'] = []
                user_acts = []
                for user_act in DA:
                    if user_act != 'Booking-Book':
                        user_acts.append(user_act)
                for user_act in user_acts:
                    del DA[user_act]

        if DA == {}:
            return {'general-greet': [['none', 'none']]}
        return DA

    def _update_state(self, user_act=None):
        if not isinstance(user_act, dict):
            raise Exception('Expect user_act to be <class \'dict\'> type but get {}.'.format(type(user_act)))
        previous_state = self.last_state
        new_belief_state = copy.deepcopy(previous_state['belief_state'])
        new_request_state = copy.deepcopy(previous_state['request_state'])
        for domain_type in user_act.keys():
            domain, tpe = domain_type.lower().split('-')
            if domain in ['unk', 'general', 'booking']:
                continue
            if tpe == 'inform':
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if k is None:
                        continue
                    try:
                        assert domain in new_belief_state
                    except:
                        raise Exception('Error: domain <{}> not in new belief state'.format(domain))
                    domain_dic = new_belief_state[domain]
                    assert 'semi' in domain_dic
                    assert 'book' in domain_dic

                    if k in domain_dic['semi']:
                        nvalue =  v
                        new_belief_state[domain]['semi'][k] = nvalue
                    elif k in domain_dic['book']:
                        new_belief_state[domain]['book'][k] = v
                    elif k.lower() in domain_dic['book']:
                        new_belief_state[domain]['book'][k.lower()] = v
                    elif k == 'trainID' and domain == 'train':
                        new_belief_state[domain]['book'][k] = v
                    else:
                        # raise Exception('unknown slot name <{}> of domain <{}>'.format(k, domain))
                        with open('unknown_slot.log', 'a+') as f:
                            f.write('unknown slot name <{}> of domain <{}>\n'.format(k, domain))
            elif tpe == 'request':
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if domain not in new_request_state:
                        new_request_state[domain] = {}
                    if k not in new_request_state[domain]:
                        new_request_state[domain][k] = 0

        new_state = copy.deepcopy(previous_state)
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        new_state['user_action'] = user_act
        
        return new_state


    def _update_greeting(self, user_act, state, DA):
        """ General request / inform. """
        _, intent_type = user_act.split('-')

        # Respond to goodbye
        if intent_type == 'bye':
            if 'general-bye' not in DA:
                DA['general-bye'] = []
            if random.random() < 0.3:
                if 'general-welcome' not in DA:
                    DA['general-welcome'] = []
        elif intent_type == 'thank':
            DA['general-welcome'] = []

    def _book_taxi(self, user_act, state, DA):
        """ Book a taxi for user. """

        blank_info = []
        for info in ['departure', 'destination']:
            if state['belief_state']['taxi']['semi'] == "":
                info = REF_USR_DA['Taxi'].get(info, info)
                blank_info.append(info)
        if state['belief_state']['taxi']['semi']['leaveAt'] == "" and state['belief_state']['taxi']['semi']['arriveBy'] == "":
            blank_info += ['Leave', 'Arrive']


        # Finish booking, tell user car type and phone number
        if len(blank_info) == 0:
            if 'Taxi-Inform' not in DA:
                DA['Taxi-Inform'] = []
            car = generate_car()
            phone_num = generate_phone_num(11)
            DA['Taxi-Inform'].append(['Car', car])
            DA['Taxi-Inform'].append(['Phone', phone_num])
            return

        # Need essential info to finish booking
        request_num = random.randint(0, 999999) % len(blank_info) + 1
        if 'Taxi-Request' not in DA:
            DA['Taxi-Request'] = []
        for i in range(request_num):
            slot = REF_USR_DA.get(blank_info[i], blank_info[i])
            DA['Taxi-Request'].append([slot, '?'])

    def _update_booking(self, user_act, state, DA):
        pass

    def _update_DA(self, user_act, user_action, state, DA):
        """ Answer user's utterance about any domain other than taxi or train. """

        domain, intent_type = user_act.split('-')

        constraints = []
        for slot in state['belief_state'][domain.lower()]['semi']:
            if state['belief_state'][domain.lower()]['semi'][slot] != "":
                constraints.append([slot, state['belief_state'][domain.lower()]['semi'][slot]])

        kb_result = self.db.query(domain.lower(), constraints)
        self.kb_result[domain] = deepcopy(kb_result)

        # Respond to user's request
        if intent_type == 'Request':
            if self.recommend_flag > 1:
                self.recommend_flag = -1
                self.choice = ""
            elif self.recommend_flag == 1:
                self.recommend_flag == 0
            if (domain + "-Inform") not in DA:
                DA[domain + "-Inform"] = []
            for slot in user_action[user_act]:
                if len(kb_result) > 0:
                    kb_slot_name = REF_SYS_DA[domain].get(slot[0], slot[0])
                    if kb_slot_name in kb_result[0]:
                        DA[domain + "-Inform"].append([slot[0], kb_result[0][kb_slot_name]])
                    else:
                        DA[domain + "-Inform"].append([slot[0], "unknown"])

        else:
            # There's no result matching user's constraint
            if len(kb_result) == 0:
                if (domain + "-NoOffer") not in DA:
                    DA[domain + "-NoOffer"] = []

                for slot in state['belief_state'][domain.lower()]['semi']:
                    if state['belief_state'][domain.lower()]['semi'][slot] != "" and \
                            state['belief_state'][domain.lower()]['semi'][slot] != "do n't care":
                        slot_name = REF_USR_DA[domain].get(slot, slot)
                        DA[domain + "-NoOffer"].append([slot_name, state['belief_state'][domain.lower()]['semi'][slot]])

                p = random.random()

                # Ask user if he wants to change constraint
                if p < 0.3:
                    req_num = min(random.randint(0, 999999) % len(DA[domain + "-NoOffer"]) + 1, 3)
                    if domain + "-Request" not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        slot_name = REF_USR_DA[domain].get(DA[domain + "-NoOffer"][i][0], DA[domain + "-NoOffer"][i][0])
                        DA[domain + "-Request"].append([slot_name, "?"])

            # There's exactly one result matching user's constraint
            elif len(kb_result) == 1:

                # Inform user about this result
                if (domain + "-Inform") not in DA:
                    DA[domain + "-Inform"] = []
                props = []
                for prop in state['belief_state'][domain.lower()]['semi']:
                    props.append(prop)
                property_num = len(props)
                if property_num > 0:
                    info_num = random.randint(0, 999999) % property_num + 1
                    random.shuffle(props)
                    for i in range(info_num):
                        slot_name = REF_USR_DA[domain].get(props[i], props[i])
                        DA[domain + "-Inform"].append([slot_name, kb_result[0][props[i]]])

            # There are multiple resultes matching user's constraint
            else:
                p = random.random()

                # Recommend a choice from kb_list
                if True: #p < 0.3:
                    if (domain + "-Inform") not in DA:
                        DA[domain + "-Inform"] = []
                    if (domain + "-Recommend") not in DA:
                        DA[domain + "-Recommend"] = []
                    DA[domain + "-Inform"].append(["Choice", str(len(kb_result))])
                    idx = random.randint(0, 999999) % len(kb_result)
                    choice = kb_result[idx]
                    if domain in ["Hotel", "Attraction", "Police", "Restaurant"]:
                        DA[domain + "-Recommend"].append(['Name', choice['name']])
                    self.recommend_flag = 0
                    self.candidate = choice
                    props = []
                    for prop in choice:
                        props.append([prop, choice[prop]])
                    prop_num = min(random.randint(0, 999999) % 3, len(props))
                    random.shuffle(props)
                    for i in range(prop_num):
                        slot = props[i][0]
                        string = REF_USR_DA[domain].get(slot, slot)
                        if string in INFORMABLE_SLOTS:
                            DA[domain + "-Recommend"].append([string, str(props[i][1])])

                # Ask user to choose a candidate.
                elif p < 0.5:
                    prop_values = []
                    props = []
                    for prop in kb_result[0]:
                        for candidate in kb_result:
                            if prop not in candidate:
                                continue
                            if candidate[prop] not in prop_values:
                                prop_values.append(candidate[prop])
                        if len(prop_values) > 1:
                            props.append([prop, prop_values])
                        prop_values = []
                    random.shuffle(props)
                    idx = 0
                    while idx < len(props):
                        if props[idx][0] not in SELECTABLE_SLOTS[domain]:
                            props.pop(idx)
                            idx -= 1
                        idx += 1
                    if domain + "-Select" not in DA:
                        DA[domain + "-Select"] = []
                    for i in range(min(len(props[0][1]), 5)):
                        prop_value = REF_USR_DA[domain].get(props[0][0], props[0][0])
                        DA[domain + "-Select"].append([prop_value, props[0][1][i]])

                # Ask user for more constraint
                else:
                    reqs = []
                    for prop in state['belief_state'][domain.lower()]['semi']:
                        if state['belief_state'][domain.lower()]['semi'][prop] == "":
                            prop_value = REF_USR_DA[domain].get(prop, prop)
                            reqs.append([prop_value, "?"])
                    i = 0
                    while i < len(reqs):
                        if reqs[i][0] not in REQUESTABLE_SLOTS:
                            reqs.pop(i)
                            i -= 1
                        i += 1
                    random.shuffle(reqs)
                    if len(reqs) == 0:
                        return
                    req_num = min(random.randint(0, 999999) % len(reqs) + 1, 2)
                    if (domain + "-Request") not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        req = reqs[i]
                        req[0] = REF_USR_DA[domain].get(req[0], req[0])
                        DA[domain + "-Request"].append(req)

    def _update_train(self, user_act, user_action, state, DA):
        constraints = []
        for time in ['leaveAt', 'arriveBy']:
            if state['belief_state']['train']['semi'][time] != "":
                constraints.append([time, state['belief_state']['train']['semi'][time]])

        if len(constraints) == 0:
            p = random.random()
            if 'Train-Request' not in DA:
                DA['Train-Request'] = []
            if p < 0.33:
                DA['Train-Request'].append(['Leave', '?'])
            elif p < 0.66:
                DA['Train-Request'].append(['Arrive', '?'])
            else:
                DA['Train-Request'].append(['Leave', '?'])
                DA['Train-Request'].append(['Arrive', '?'])

        if 'Train-Request' not in DA:
            DA['Train-Request'] = []
        for prop in ['day', 'destination', 'departure']:
            if state['belief_state']['train']['semi'][prop] == "":
                slot = REF_USR_DA['Train'].get(prop, prop)
                DA["Train-Request"].append([slot, '?'])
            else:
                constraints.append([prop, state['belief_state']['train']['semi'][prop]])

        kb_result = self.db.query('train', constraints)
        self.kb_result['Train'] = deepcopy(kb_result)

        if user_act == 'Train-Request':
            del(DA['Train-Request'])
            if 'Train-Inform' not in DA:
                DA['Train-Inform'] = []
            for slot in user_action[user_act]:
                slot_name = REF_SYS_DA['Train'].get(slot[0], slot[0])
                try:
                    DA['Train-Inform'].append([slot[0], kb_result[0][slot_name]])
                except:
                    pass
            return
        if len(kb_result) == 0:
            if 'Train-NoOffer' not in DA:
                DA['Train-NoOffer'] = []
            for prop in constraints:
                DA['Train-NoOffer'].append([REF_USR_DA['Train'].get(prop[0], prop[0]), prop[1]])
            if 'Train-Request' in DA:
                del DA['Train-Request']
        elif len(kb_result) >= 1:
            if len(constraints) < 4:
                return
            if 'Train-Request' in DA:
                del DA['Train-Request']
            if 'Train-OfferBook' not in DA:
                DA['Train-OfferBook'] = []
            for prop in constraints:
                DA['Train-OfferBook'].append([REF_USR_DA['Train'].get(prop[0], prop[0]), prop[1]])

    def _judge_booking(self, user_act, user_action, DA):
        """ If user want to book, return a ref number. """
        if self.recommend_flag > 1:
            self.recommend_flag = -1
            self.choice = ""
        elif self.recommend_flag == 1:
            self.recommend_flag == 0
        domain, _ = user_act.split('-')
        for slot in user_action[user_act]:
            if domain in booking_info and slot[0] in booking_info[domain]:
                if 'Booking-Book' not in DA:
                    if domain in self.kb_result and len(self.kb_result[domain]) > 0:
                        if 'Ref' in self.kb_result[domain][0]:
                            DA['Booking-Book'] = [["Ref", self.kb_result[domain][0]['Ref']]]
                        else:
                            DA['Booking-Book'] = [["Ref", "N/A"]]
        # TODO handle booking between multi turn

def check_diff(last_state, state):
    user_action = {}
    if last_state == {}:
        for domain in state['belief_state']:
            for slot in state['belief_state'][domain]['book']:
                if slot != 'booked' and state['belief_state'][domain]['book'][slot] != '':
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot), state['belief_state'][domain]['book'][slot]] \
                            not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append([REF_USR_DA[domain.capitalize()].get(slot, slot), \
                                                                    state['belief_state'][domain]['book'][slot]])
            for slot in state['belief_state'][domain]['semi']:
                if state['belief_state'][domain]['semi'][slot] != "":
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot), state['belief_state'][domain]['semi'][slot]] \
                            not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append([REF_USR_DA[domain.capitalize()].get(slot, slot), \
                                                                    state['belief_state'][domain]['semi'][slot]])
        for domain in state['request_state']:
            for slot in state['request_state'][domain]:
                if (domain.capitalize() + "-Request") not in user_action:
                    user_action[domain.capitalize() + "-Request"] = []
                if [REF_USR_DA[domain].get(slot, slot), '?'] not in user_action[domain.capitalize() + "-Request"]:
                    user_action[domain.capitalize() + "-Request"].append([REF_USR_DA[domain].get(slot, slot), '?'])

    else:
        for domain in state['belief_state']:
            for slot in state['belief_state'][domain]['book']:
                if slot != 'booked' and state['belief_state'][domain]['book'][slot] != last_state['belief_state'][domain]['book'][slot]:
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot),
                        state['belief_state'][domain]['book'][slot]] \
                            not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append(
                            [REF_USR_DA[domain.capitalize()].get(slot, slot), \
                             state['belief_state'][domain]['book'][slot]])
            for slot in state['belief_state'][domain]['semi']:
                if state['belief_state'][domain]['semi'][slot] != last_state['belief_state'][domain]['semi'][slot] and \
                        state['belief_state'][domain]['semi'][slot] != '':
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot), state['belief_state'][domain]['semi'][slot]] \
                        not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append([REF_USR_DA[domain.capitalize()].get(slot, slot), \
                                                                    state['belief_state'][domain]['semi'][slot]])
        for domain in state['request_state']:
            for slot in state['request_state'][domain]:
                if (domain not in last_state['request_state']) or (slot not in last_state['request_state'][domain]):
                    if (domain.capitalize() + "-Request") not in user_action:
                        user_action[domain.capitalize() + "-Request"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot), '?'] not in user_action[domain.capitalize() + "-Request"]:
                        user_action[domain.capitalize() + "-Request"].append([REF_USR_DA[domain.capitalize()].get(slot, slot), '?'])
    return user_action


def deduplicate(lst):
    i = 0
    while i < len(lst):
        if lst[i] in lst[0 : i]:
            lst.pop(i)
            i -= 1
        i += 1
    return lst

def generate_phone_num(length):
    """ Generate a phone num. """
    string = ""
    while len(string) < length:
        string += digit[random.randint(0, 999999) % 10]
    return string

def generate_car():
    """ Generate a car for taxi booking. """
    car_types = ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"]
    p = random.randint(0, 999999) % len(car_types)
    return car_types[p]

def init_belief_state():
    belief_state = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    }
    state = {'user_action': {},
     'belief_state': belief_state,
     'request_state': {}}
    return state
