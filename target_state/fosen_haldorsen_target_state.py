    #ideal states are defined externally
    with open('init_state//fosen_haldorsen//station.json', 'r') as f:
        ideal_state_json = json.load(f)

    #a bit arbitrary how to set up stations
    for st in stations:
        if st.id in ideal_state_json.keys():
            ideal = {int(k): int(v) for k, v in ideal_state_json[st.id].items()}
            st.ideal_state = ideal
        else:
            ideal = {}
            for hour in range(0, 24):
                ideal[hour] = st.station_cap // 2
            st.ideal_state = ideal

        st.current_charged_bikes = min(st.station_cap, st.actual_num_bikes[init_hour])
        st.current_flat_bikes = 0
        st.init_charged = st.current_charged_bikes
        st.init_flat = st.current_flat_bikes

