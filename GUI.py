import dearpygui.dearpygui as dpg

from utilities import m

dpg.create_context()
WIDTH = 1000
HEIGHT = 800

btn_α_id = 0
btn_c_id = 0
btn_eps_id = 0
btn_k_id = 0
btn_R_id = 0

plt1 = []
plt2 = []


def send_items():
    α = dpg.get_value(btn_α_id)
    eps = 10 ** dpg.get_value(btn_eps_id)
    c = dpg.get_value(btn_c_id)
    k = dpg.get_value(btn_k_id)
    R = dpg.get_value(btn_R_id)
    l = 12
    T = 250
    [r_x, x_l, r_t, t_l] = m(α, c, l, T, k, R, eps)
    count = 0
    for key, value in r_x.items():
        dpg.set_value(plt1[count], [x_l, value])
        dpg.set_item_label(plt1[count], label="t=" + str(key) + " c")
        count = count + 1
    count = 0
    for key, value in r_t.items():
        dpg.set_value(plt2[count], [t_l, value])
        dpg.set_item_label(plt2[count], label="x=" + str(key) + " cm")
        count = count + 1


with dpg.window(label="Enter params", pos=(0, 0), height=round(HEIGHT / 3), width=round(WIDTH / 3)) as win:
    btn_eps_id = dpg.add_input_int(label="eps 10^(-x)", enabled=True, default_value=-5)
    btn_α_id = dpg.add_input_double(label="a", enabled=True, default_value=0.002, format="%.4f")
    btn_c_id = dpg.add_input_double(label="c", enabled=True, default_value=1.65, format="%.4f")
    btn_k_id = dpg.add_input_double(label="k", enabled=True, default_value=0.59, format="%.4f")
    btn_R_id = dpg.add_input_double(label="R", enabled=True, default_value=0.1, format="%.4f")

    button_calculate = dpg.add_button(label="Calculate", callback=send_items)

with dpg.window(label="Tutorial", tag="win", pos=(round(HEIGHT / 2), 0)):
    [r_x, x_l, r_t, t_l] = m(0.002, 1.65, 12, 250, 0.59, 0.1, 0.00001)
    with dpg.plot(label="plt1", height=round(WIDTH / 2), width=round(HEIGHT - HEIGHT / 3)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="coordinate X, cm")
        dpg.add_plot_axis(dpg.mvYAxis, label="Temperature w, K", tag="y_axis1")
        # α, c, l, T, k, R, eps

        for key, value in r_x.items():
            tag = dpg.generate_uuid()
            plt1.append(tag)
            dpg.add_line_series(x_l, value, label="t=" + str(key) + " c", parent="y_axis1", tag=tag)
    with dpg.plot(label="plt2", height=round(WIDTH / 2), width=round(HEIGHT - HEIGHT / 3)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="time T, c")
        dpg.add_plot_axis(dpg.mvYAxis, label="Temperature w, K", tag="y_axis2")

        for key, value in r_t.items():
            tag = dpg.generate_uuid()
            plt2.append(tag)
            dpg.add_line_series(t_l, value, label="x=" + str(key) + " cm", parent="y_axis2", tag=tag)

dpg.create_viewport(title='Custom Title', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
