import altair as alt

def get_chart(data):
    hover = alt.selection_single(
        fields=["Time"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Cure cycle")
        .mark_line()
        .encode(
            x="Time",
            y= alt.Y("Temperature", scale=alt.Scale(domain=[270, 520])),
            color=alt.Color("method", scale=alt.
                    Scale(domain=['T Air' ,'T Pre'], range=['black', 'blue']))
            # strokeDash="method",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="Time",
            y="Temperature",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Time", title="Time (min)"),
                alt.Tooltip("Temperature", title="Temperature (k)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()