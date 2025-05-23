from enigma import *



# Code Crack 1 

plugboard_1 = Plugboard()
plugboard_1.add(PlugLead('KI'))
plugboard_1.add(PlugLead('XN'))
plugboard_1.add(PlugLead('FL'))
ring_settings_1 = [4, 2, 14]
rotor_adjuster_1 = Rotor_Adjuster(ring_settings_1)
reflector_1 = Reflector('C', rotor_adjuster_1)
rotor_engine_1 = Rotor_Engine(rotor_adjuster_1, reflector_1)
rotor_engine_1.add_rotor(Rotor('Beta'))
rotor_engine_1.add_rotor(Rotor('Gamma'))
rotor_engine_1.add_rotor(Rotor('V'))
reflector_1.set_rotor_count(rotor_engine_1.get_rotors_count())
rotor_turner_1 = Rotor_Turner(ring_settings_1, rotor_adjuster_1)
notch_adjuster_1 = Notch_Adjuster(ring_settings_1, rotor_engine_1, rotor_adjuster_1)
enigma_1 = EnigmaMachine(['M', 'J', 'M'], rotor_engine_1, rotor_turner_1, notch_adjuster_1, rotor_adjuster_1, reflector_1, plugboard_1)

crib_1 = 'SECRETS'
reflector_options_1 = ['A', 'B', 'C']
encoded_message_1 = 'DMEXBMKYCVPNQBEDHXVPZGKMTFFBJRPJTLHLCHOTKOYXGGHZ'

def decode_1(encoded_message, reflector_options, crib):
    for reflector_type in reflector_options:
        enigma_1.reset_position_settings(['M', 'J', 'M'])
        reflector_1.set_reflector_type(reflector_type)
        decoded_message = enigma_1.encode(encoded_message, True)
        if crib in decoded_message:
            print(decoded_message)
            print('Reflector', reflector_type)
            break

decode_1(encoded_message_1, reflector_options_1, crib_1)




# Code Crack 2

plugboard_2 = Plugboard()
plugboard_2.add(PlugLead('VH'))
plugboard_2.add(PlugLead('PT'))
plugboard_2.add(PlugLead('ZG'))
plugboard_2.add(PlugLead('BJ'))
plugboard_2.add(PlugLead('EY'))
plugboard_2.add(PlugLead('FS'))

ring_settings_2 = [23, 2, 10]
rotor_adjuster_2 = Rotor_Adjuster(ring_settings_2)
reflector_2 = Reflector('B', rotor_adjuster_2)
rotor_engine_2 = Rotor_Engine(rotor_adjuster_2, reflector_2)
rotor_engine_2.add_rotor(Rotor('Beta'))
rotor_engine_2.add_rotor(Rotor('I'))
rotor_engine_2.add_rotor(Rotor('III'))
reflector_2.set_rotor_count(rotor_engine_2.get_rotors_count())
rotor_turner_2 = Rotor_Turner(ring_settings_2, rotor_adjuster_2)
notch_adjuster_2 = Notch_Adjuster(ring_settings_2, rotor_engine_2, rotor_adjuster_2)
enigma_2 = EnigmaMachine(['A', 'A', 'A'], rotor_engine_2, rotor_turner_2, notch_adjuster_2, rotor_adjuster_2, reflector_2, plugboard_2)

crib_2 = 'UNIVERSITY'
starting_positions_2 = ['A', 'A', 'A']
encoded_message_2 = 'CMFSUPKNCBMUYEQVVDYKLRQZTPUFHSWWAKTUGXMPAMYAFITXIJKMH'

def decode_2(encoded_message, starting_positions, crib):
    for i in range(25):
        starting_positions[0] = EnigmaConfig.enigma_data['Alphabet'][i]
        for x in range(25):
            starting_positions[1] = EnigmaConfig.enigma_data['Alphabet'][x]
            for y in range(25):
                starting_positions[2] = EnigmaConfig.enigma_data['Alphabet'][y]
                enigma_2.reset_position_settings(starting_positions)
                decoded_message = enigma_2.encode(encoded_message, True)
                if crib in decoded_message:
                    print(decoded_message)
                    print('Position Setting: ', starting_positions)
                    break

decode_2(encoded_message_2, starting_positions_2, crib_2)




# Code Crack 3

import itertools

plugboard_3 = Plugboard()
plugboard_3.add(PlugLead('FH'))
plugboard_3.add(PlugLead('TS'))
plugboard_3.add(PlugLead('BE'))
plugboard_3.add(PlugLead('UQ'))
plugboard_3.add(PlugLead('KD'))
plugboard_3.add(PlugLead('AL'))

ring_settings_3 = [2, 2, 2]
rotor_adjuster_3 = Rotor_Adjuster(ring_settings_3)
reflector_3 = Reflector('A', rotor_adjuster_3)
rotor_engine_3 = Rotor_Engine(rotor_adjuster_3, reflector_3)
reflector_3.set_rotor_count(rotor_engine_3.get_rotors_count())
rotor_turner_3 = Rotor_Turner(ring_settings_3, rotor_adjuster_3)
notch_adjuster_3 = Notch_Adjuster(ring_settings_3, rotor_engine_3, rotor_adjuster_3)
enigma_3 = EnigmaMachine(['E', 'M', 'Y'], rotor_engine_3, rotor_turner_3, notch_adjuster_3, rotor_adjuster_3, reflector_3, plugboard_3)

crib_3 = 'THOUSANDS'
encoded_message_3 = 'ABSKJAKKMRITTNYURBJFWQGRSGNNYJSDRYLAPQWIAGKJYEPCTAGDCTHLCDRZRFZHKNRSDLNPFPEBVESHPY'
ring_setting_options_3 = [2, 4, 6, 8, 10, 20, 22, 24, 26]
rotar_options_3 = ['II', 'IV', 'Beta', 'Gamma']
reflector_options_3 = ['A', 'B', 'C']

def decoder_3(encoded_message, crib, plugboard, starting_positions, ring_setting_options, rotar_options, reflector_options):
    rotar_permutations = list(itertools.product(rotar_options, repeat=3))
    reflector_permutations = reflector_options 
    ring_permutations = list(itertools.product(ring_setting_options, repeat=3))
    for rotar_permutation in range(23): 
        for rotors in rotar_permutations:
            rotor_engine_3.reset_rotors()
            rotor_engine_3.add_rotor(Rotor(rotors[0]))
            rotor_engine_3.add_rotor(Rotor(rotors[1]))
            rotor_engine_3.add_rotor(Rotor(rotors[2]))
            for reflector in reflector_permutations:
                reflector_3.set_reflector_type(reflector)
                enigma_3.set_reflector(reflector_3)
                rotor_engine_3.set_reflector(reflector_3)
                for ring_settings in ring_permutations:
                    rotor_adjuster_3.set_ring_settings(ring_settings)
                    rotor_turner_3.set_ring_settings(ring_settings)
                    notch_adjuster_3.set_ring_settings(ring_settings)
                    decoded_message = enigma_3.encode(encoded_message, True)
                    if crib in decoded_message:
                        print(decoded_message)
                        print('Ring Settings: ', ring_settings)
                        print('Rotors: ', rotors)
                        print('Reflector: ', reflector)
                        break

decoder_3(encoded_message_3, crib_3, plugboard_3, starting_positions_3, ring_setting_options_3, rotar_options_3, reflector_options_3)




# Code Crack 4

plugboard_4 = Plugboard()
plugboard_4.add(PlugLead('WP'))
plugboard_4.add(PlugLead('RJ'))
plugboard_4.add(PlugLead('VF'))
plugboard_4.add(PlugLead('HN'))
plugboard_4.add(PlugLead('CG'))
plugboard_4.add(PlugLead('BS'))

ring_settings_4 = [24, 12, 10]
rotor_adjuster_4 = Rotor_Adjuster(ring_settings_4)
reflector_4 = Reflector('A', rotor_adjuster_4)
rotor_engine_4 = Rotor_Engine(rotor_adjuster_4, reflector_4)
rotor_engine_4.add_rotor(Rotor('V'))
rotor_engine_4.add_rotor(Rotor('III'))
rotor_engine_4.add_rotor(Rotor('IV'))
reflector_4.set_rotor_count(rotor_engine_4.get_rotors_count())
rotor_turner_4 = Rotor_Turner(ring_settings_4, rotor_adjuster_4)
notch_adjuster_4 = Notch_Adjuster(ring_settings_4, rotor_engine_4, rotor_adjuster_4)
enigma_4 = EnigmaMachine(['S', 'W', 'U'], rotor_engine_4, rotor_turner_4, notch_adjuster_4, rotor_adjuster_4, reflector_4, plugboard_4)

crib_4 = 'TUTOR'
encoded_message_4 = 'SDNTVTPHRBNWTLMZTQKZGADDQYPFNHBPNHCQGBGMZPZLUAVGDQVYRBFYYEIXQWVTHXGNW'
letter_options = ['D', 'E', 'K', 'L', 'M', 'O', 'Q', 'T', 'U', 'X', 'Y', 'Z']

def decoder_4(encoded_message, crib, letter_options):
    for plug_idx_1 in range(12):
        plug_1 = letter_options[plug_idx_1]
        for plug_idx_2 in range(11):
            temp_letter_options = letter_options.copy()
            del temp_letter_options[plug_idx_1]
            plug_2 = temp_letter_options[plug_idx_2]
            plugboard_4.add(PlugLead('A' + plug_1))
            plugboard_4.add(PlugLead('I' + plug_2))
            decoded_message = enigma_4.encode(encoded_message, True)
            enigma_4.reset_position_settings(['S', 'W', 'U'])
            plugboard_4.delete()
            if crib in decoded_message:
                print(decoded_message)
                print('Pluglead Letter 1: ', plug_1)
                print('Pluglead Letter 2: ', plug_2)
                break

decoder_4(encoded_message_4, crib_4, letter_options)




# Code Crack 5

plugboard_5 = Plugboard()
plugboard_5.add(PlugLead('UG'))
plugboard_5.add(PlugLead('IE'))
plugboard_5.add(PlugLead('PO'))
plugboard_5.add(PlugLead('NX'))
plugboard_5.add(PlugLead('WT'))

ring_settings_5 = [6, 18, 7]
rotor_adjuster_5 = Rotor_Adjuster(ring_settings_5)
reflector_5 = Reflector('A', rotor_adjuster_5)
rotor_engine_5 = Rotor_Engine(rotor_adjuster_5, reflector_5)
rotor_engine_5.add_rotor(Rotor('V'))
rotor_engine_5.add_rotor(Rotor('II'))
rotor_engine_5.add_rotor(Rotor('IV'))
reflector_5.set_rotor_count(rotor_engine_5.get_rotors_count())
rotor_turner_5 = Rotor_Turner(ring_settings_5, rotor_adjuster_5)
notch_adjuster_5 = Notch_Adjuster(ring_settings_5, rotor_engine_5, rotor_adjuster_5)
enigma_5 = EnigmaMachine(['A', 'J', 'L'], rotor_engine_5, rotor_turner_5, notch_adjuster_5, rotor_adjuster_5, reflector_5, plugboard_5)

cribs = ['FACEBOOK', 'INSTAGRAM', 'LINKEDIN', 'INSTA', 'TWITTER', 'SNAPCHAT', 'FACE', 'WECHAT', 'MYSPACE', 'GRAM', 'BEBO', 'TIKTOK', 'REDDIT', 'PINTEREST', 'YOUTUBE', 'WHATSAPP', 'TUMBLR', 'TELEGRAM', 'VIBER', 'LINE', 'BAIDU', 'SINA', 'QZONE', 'DOUBAN', 'TWITCH', 'SIGNAL', 'BUMBLE', 'DISCORD', 'HINGE', 'QUORA', 'FLICKR', 'NEXTDOOR', 'SIGNAL', 'MESSENGER', 'DOUYIN', 'KUAISHOU', 'WEIBOU', 'TEAMS', 'TIEBA', 'SKYPE']
encoded_message_5 = 'HWREISXLGTTBYVXRCWWJAKZDTVZWKBDJPVQYNEQIOTIFX'
standard_reflectors = [EnigmaConfig.enigma_data['A'], EnigmaConfig.enigma_data['B'] , EnigmaConfig.enigma_data['C']]

def swap_elements(list_, pos1, pos2):
    list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
    
def decoder_5(encoded_message, cribs, standard_reflectors):
    for original_reflector in standard_reflectors:
        enigma_5.reset_position_settings(['A', 'J', 'L'])
        possible_swaps = itertools.combinations(range(26), 4)
        for swap in possible_swaps:
            modified_reflector = original_reflector.copy()
            swap_elements(modified_reflector, swap[0], swap[1])
            swap_elements(modified_reflector, swap[2], swap[3])
            reflector_5.set_reflector_type(modified_reflector)
            enigma_5.set_reflector(reflector_5)
            rotor_engine_5.set_reflector(reflector_5)
            decoded_message = enigma_5.encode(encoded_message, True)
            for crib in cribs:
                if crib in decoded_message:
                    print(decoded_message)
                    print('Reflector: ', modified_reflector)
            

decoder_5(encoded_message_5, cribs, standard_reflectors)

