from sting.shunt.parallel_rc import ShuntParallelRC

def combine_shunts(system):

    print("> Reduce shunts to have one shunt per bus:")

    shunt_df = (
        system
        .view("shunts", attrs=["bus_idx", "g", "b"], dataframe=True)
        .reset_index(drop=True)
        .pivot_table(index="bus_idx", values=["g", "b"], aggfunc="sum")
    )

    shunt_df["r"] = 1 / shunt_df["g"]
    shunt_df["c"] = 1 / shunt_df["b"]
    shunt_df["idx"] = range(len(shunt_df))
    shunt_df.drop(columns=["b", "g"], inplace=True)

    # Clear all existing parallel RC shunts
    system.clear("pa_rc")

    # Add each effective/combined parallel RC shunt to the pa_rc components
    for _, row in shunt_df.iterrows():
        shunt = ShuntParallelRC(**row.to_dict())
        system.add(shunt)

    print("\t- New list of parallel RC components created ... ok\n")
