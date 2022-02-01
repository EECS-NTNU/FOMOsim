"""
General methods for exporting results for evaluation to excel
"""
import analysis.evaluate_policies
import classes
from globals import HyperParameters, EXCEL_EXPORT_DIR
import pandas as pd
import os
from openpyxl import load_workbook


def metrics_to_xlsx(instances: [classes.World]):
    """
    Method to export metrics for a list of evaluated instances to Excel
    :param instances: list of instances to be exported
    """

    # assuming that all instances is tested on the same parameter
    base_instance = instances[0]
    parameter_name = (
        base_instance.metrics.testing_parameter_name
        if base_instance.metrics.testing_parameter_name != ""
        else "TEST"
    )
    # list to handle the column name structure
    column_tuples = []
    # list for all metrics data
    metrics_data = []

    # loop through all instances and record their metrics
    for instance in instances:
        metrics_data, column_tuples = add_metric_column(
            instance, column_tuples=column_tuples, metrics_data=metrics_data
        )

    # creating the directory to save the file if it doesn't exist
    if not os.path.exists(EXCEL_EXPORT_DIR):
        os.makedirs(EXCEL_EXPORT_DIR)

    file_name = f"{EXCEL_EXPORT_DIR}/{parameter_name.title()}.xlsx"

    # if the file isn't created -> create a new .xlsx file
    if not os.path.isfile(file_name):
        pd.DataFrame().to_excel(file_name)
    # load the current excel file
    book = load_workbook(file_name)
    # creating a writer
    writer = pd.ExcelWriter(file_name, engine="openpyxl")
    # adds the current book/file to the writer and adding the existing sheets (else the writer will overwrite)
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    # creating a list of existing sheets, so that if there are other sheets with a same name, a suffix is added
    sheets = [ws.title.split("_")[0] for ws in book.worksheets]

    # have to replace : since its illegal that a sheet name contains the character
    instance_created_at = base_instance.created_at.replace(":", ".")

    # creating the sheet name, adding suffix if other sheets with the same name exist
    sheet_name = (
        f"{instance_created_at}_1"
        if instance_created_at not in sheets
        else f"{instance_created_at}_{sheets.count(instance_created_at) + 1}"
    )

    try:
        # creating a multiindex object so that columns with the same name gets merged
        columns = pd.MultiIndex.from_tuples(
            column_tuples,
            names=[
                "Parameter",
                "Parameter value",
                "Policy",
                "Shifts trained",
                "Metrics",
            ],
        )

        df = pd.DataFrame(
            pd.concat(metrics_data, axis=1, ignore_index=True).to_numpy(),
            columns=columns,
        )  # concatenate all metrics dataframes to one dataframe

        df.to_excel(
            writer, sheet_name=sheet_name, startcol=1, startrow=1
        )  # write dataframe to file

        # adding a sheets of all globals defined in HyperParameter
        values = []
        names = []
        for hyper_parameter in HyperParameters().__dict__.keys():
            value = getattr(base_instance, hyper_parameter)
            names.append(hyper_parameter.replace("_", " ").capitalize())
            values.append(value)

        df_globals = pd.DataFrame(values, index=names, columns=["Globals"])

        df_globals.to_excel(
            writer, sheet_name=f"{sheet_name}_g", startcol=1, startrow=1
        )

    finally:
        writer.save()
        return sheet_name


def add_metric_column(instance, metrics_data, column_tuples):
    # making a dict of all metric variables and their values
    metric_variables = instance.metrics.__dict__
    # removing testing parameter name and value -> easier to have control over all metrics and add new one
    parameter_name = metric_variables["testing_parameter_name"]
    del metric_variables["testing_parameter_name"]
    parameter_value = metric_variables["testing_parameter_value"]
    del metric_variables["testing_parameter_value"]

    for i, metric in enumerate(metric_variables.keys()):
        # making the column name hierarchy to ensure that right name is associated with right list of values
        column_tuples.append(
            (
                parameter_name.title(),
                str(parameter_value),
                instance.policy.__str__(),
                str(instance.policy.value_function.shifts_trained)
                if hasattr(instance.policy, "value_function")
                else "N.A",
                metric.replace("_", " ").title(),
            )
        )
        # adding a dataframe of the metric data
        # (have to be dataframe to be able to merge all metrics to one matrix later)
        metrics_data.append(pd.DataFrame({i: metric_variables[metric]}))

    return metrics_data, column_tuples


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[1]}")
        analysis.evaluate_policies.run_analysis_from_path(
            sys.argv[1], export_to_excel=True
        )
    else:
        analysis.evaluate_policies.run_analysis_from_path(
            "world_cache/test_models",
            shift_duration=60,
            export_to_excel=True,
        )
