import dearpygui.dearpygui as dpg

from utilities import m

dpg.create_context()

WIDTH = 1300
HEIGHT = 900

btn_α_id = 0
btn_c_id = 0
btn_eps_id = 0
btn_k_id = 0
btn_R_id = 0
btn_n_id = 0
plt1 = []
plt2 = []
table_id = 0

[r_x, x_l, r_t, t_l, z, φn] = m(10, 0.002, 1.65, 12, 250, 0.59, 0.1, 0.00001)


def clear_table(z, φ):
    for tag in dpg.get_item_children(table_id)[1]:
        dpg.delete_item(tag)

    for i in range(0, len(z)):
        with dpg.table_row(parent=table_id, tag=f'row_{i}'):
            dpg.add_text(f"{i + 1}: {format(z[i], '.6f')} {format(φ[i], '.6f')} ")


def send_items():
    n = dpg.get_value(btn_n_id)
    α = dpg.get_value(btn_α_id)
    eps = 10 ** dpg.get_value(btn_eps_id)
    c = dpg.get_value(btn_c_id)
    k = dpg.get_value(btn_k_id)
    R = dpg.get_value(btn_R_id)
    l = 12
    T = 250
    try:
        [r_x, x_l, r_t, t_l, zn, φn] = m(n, α, c, l, T, k, R, eps)
    except:
        return

    z = zn
    φ = φn
    clear_table(z, φ)
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


with dpg.window(label="Enter params", pos=(0, 0), height=round(HEIGHT / 2 - 20), width=round(WIDTH / 4) - 2) as win:
    btn_eps_id = dpg.add_input_int(label="eps 10^(-x)", enabled=True, default_value=-5)
    btn_α_id = dpg.add_input_double(label="a", enabled=True, default_value=0.002, format="%.4f", step=0.01)
    btn_c_id = dpg.add_input_double(label="c", enabled=True, default_value=1.65, format="%.4f", step=0.01)
    btn_k_id = dpg.add_input_double(label="k", enabled=True, default_value=0.59, format="%.4f", step=0.01)
    btn_R_id = dpg.add_input_double(label="R", enabled=True, default_value=0.1, format="%.4f", step=0.01)
    btn_n_id = dpg.add_input_int(label="n", enabled=True, default_value=10)

    dpg.set_item_callback(btn_eps_id, send_items)
    dpg.set_item_callback(btn_α_id, send_items)
    dpg.set_item_callback(btn_c_id, send_items)
    dpg.set_item_callback(btn_k_id, send_items)
    dpg.set_item_callback(btn_R_id, send_items)
    dpg.set_item_callback(btn_n_id, send_items)

with dpg.window(label="PLOT", tag="win", pos=(round(WIDTH / 4), 0), width=round(3 * WIDTH / 4) - 37):
    with dpg.plot(label="plt1", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="coordinate X, cm", tag="x_axis1")
        dpg.add_plot_axis(dpg.mvYAxis, label="Temperature w, K", tag="y_axis1")
        for key, value in r_x.items():
            tag = dpg.generate_uuid()
            plt1.append(tag)
            dpg.add_line_series(x_l, value, label="t=" + str(key) + " c", parent="y_axis1", tag=tag)
    with dpg.plot(label="plt2", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="time T, c", tag="x_axis2")
        dpg.add_plot_axis(dpg.mvYAxis, label="Temperature w, K", tag="y_axis2")

        for key, value in r_t.items():
            tag = dpg.generate_uuid()
            plt2.append(tag)
            dpg.add_line_series(t_l, value, label="x=" + str(key) + " cm", parent="y_axis2", tag=tag)

with dpg.window(label="N,    z,    fi", pos=(round(2 * WIDTH), 0)):
    table_id = dpg.generate_uuid()
    with dpg.table(header_row=False, width=600, height=HEIGHT, tag=table_id):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()

        for i in range(0, len(z)):
            with dpg.table_row(parent=table_id, tag=f'row_{i}'):
                dpg.add_text(f"  {i + 1}: {format(z[i], '.6f')} {format(φn[i], '.6f')} ")

dpg.create_viewport(title='Modeling thermal process', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
