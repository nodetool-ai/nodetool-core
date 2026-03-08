import { BaseNode } from "@nodetool/node-sdk";

type DateValue = { year: number; month: number; day: number };
type DateTimeValue = {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  millisecond: number;
  tzinfo?: string;
  utc_offset?: string;
};

type DateFormat =
  | "%Y-%m-%d"
  | "%m/%d/%Y"
  | "%d/%m/%Y"
  | "%B %d, %Y"
  | "%Y%m%d"
  | "%Y%m%d_%H%M%S"
  | "%Y-%m-%dT%H:%M:%S"
  | "%Y-%m-%dT%H:%M:%S%z";

type TimeDirection = "past" | "future";
type TimeUnit = "hours" | "days" | "months";
type BoundaryType = "start" | "end";
type PeriodType = "day" | "week" | "month" | "year";

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

function isDateTimeValue(value: unknown): value is DateTimeValue {
  return !!value && typeof value === "object" && "year" in (value as object) && "hour" in (value as object);
}

function isDateValue(value: unknown): value is DateValue {
  return !!value && typeof value === "object" && "year" in (value as object) && !("hour" in (value as object));
}

function toDate(input: unknown): Date {
  if (isDateTimeValue(input)) {
    const tzOffset = String(input.utc_offset ?? "");
    if (tzOffset && /^[+-]\d{2}:?\d{2}$/.test(tzOffset)) {
      const normalized = tzOffset.includes(":") ? tzOffset : `${tzOffset.slice(0, 3)}:${tzOffset.slice(3)}`;
      const iso = `${String(input.year).padStart(4, "0")}-${String(input.month).padStart(2, "0")}-${String(
        input.day
      ).padStart(2, "0")}T${String(input.hour).padStart(2, "0")}:${String(input.minute).padStart(2, "0")}:${String(
        input.second
      ).padStart(2, "0")}.${String(input.millisecond ?? 0).padStart(3, "0")}${normalized}`;
      return new Date(iso);
    }
    return new Date(
      Number(input.year),
      Number(input.month) - 1,
      Number(input.day),
      Number(input.hour),
      Number(input.minute),
      Number(input.second),
      Number(input.millisecond ?? 0)
    );
  }
  if (isDateValue(input)) {
    return new Date(Number(input.year), Number(input.month) - 1, Number(input.day), 0, 0, 0, 0);
  }
  if (typeof input === "string") {
    return new Date(input);
  }
  return new Date();
}

function toDateValue(date: Date): DateValue {
  return {
    year: date.getFullYear(),
    month: date.getMonth() + 1,
    day: date.getDate(),
  };
}

function utcOffsetString(date: Date): string {
  const offsetMin = -date.getTimezoneOffset();
  const sign = offsetMin >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMin);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  return `${sign}${hh}${mm}`;
}

function toDateTimeValue(date: Date): DateTimeValue {
  return {
    year: date.getFullYear(),
    month: date.getMonth() + 1,
    day: date.getDate(),
    hour: date.getHours(),
    minute: date.getMinutes(),
    second: date.getSeconds(),
    millisecond: date.getMilliseconds(),
    tzinfo: date.toString().match(/\(([^)]+)\)$/)?.[1] ?? "",
    utc_offset: utcOffsetString(date),
  };
}

function formatDate(date: Date, format: DateFormat): string {
  const year = String(date.getFullYear()).padStart(4, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");

  switch (format) {
    case "%Y-%m-%d":
      return `${year}-${month}-${day}`;
    case "%m/%d/%Y":
      return `${month}/${day}/${year}`;
    case "%d/%m/%Y":
      return `${day}/${month}/${year}`;
    case "%B %d, %Y":
      return `${MONTHS[date.getMonth()]} ${day}, ${year}`;
    case "%Y%m%d":
      return `${year}${month}${day}`;
    case "%Y%m%d_%H%M%S":
      return `${year}${month}${day}_${hour}${minute}${second}`;
    case "%Y-%m-%dT%H:%M:%S":
      return `${year}-${month}-${day}T${hour}:${minute}:${second}`;
    case "%Y-%m-%dT%H:%M:%S%z":
      return `${year}-${month}-${day}T${hour}:${minute}:${second}${utcOffsetString(date)}`;
    default:
      return date.toISOString();
  }
}

function parseDateByFormat(value: string, format: DateFormat): Date {
  const s = value.trim();
  let m: RegExpMatchArray | null;

  if (format === "%Y-%m-%d" && (m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/))) {
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  }
  if (format === "%m/%d/%Y" && (m = s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/))) {
    return new Date(Number(m[3]), Number(m[1]) - 1, Number(m[2]));
  }
  if (format === "%d/%m/%Y" && (m = s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/))) {
    return new Date(Number(m[3]), Number(m[2]) - 1, Number(m[1]));
  }
  if (format === "%B %d, %Y" && (m = s.match(/^([A-Za-z]+)\s+(\d{2}),\s*(\d{4})$/))) {
    const monthToken = m[1];
    const monthIndex = MONTHS.findIndex((n) => n.toLowerCase() === monthToken.toLowerCase());
    if (monthIndex < 0) throw new Error(`Invalid date string: ${value}`);
    return new Date(Number(m[3]), monthIndex, Number(m[2]));
  }
  if (format === "%Y%m%d" && (m = s.match(/^(\d{4})(\d{2})(\d{2})$/))) {
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  }
  if (format === "%Y%m%d_%H%M%S" && (m = s.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/))) {
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6]));
  }
  if (format === "%Y-%m-%dT%H:%M:%S" && (m = s.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$/))) {
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6]));
  }
  if (
    format === "%Y-%m-%dT%H:%M:%S%z" &&
    (m = s.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]\d{2}:?\d{2})$/))
  ) {
    const tz = m[7].includes(":") ? m[7] : `${m[7].slice(0, 3)}:${m[7].slice(3)}`;
    return new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}${tz}`);
  }

  throw new Error(`Invalid date string for format ${format}: ${value}`);
}

export class TodayLibNode extends BaseNode {
  static readonly nodeType = "lib.date.Today";
  static readonly title = "Today";
  static readonly description = "Get the current date in Date format.";

  async process(): Promise<Record<string, unknown>> {
    return { output: toDateValue(new Date()) };
  }
}

export class NowLibNode extends BaseNode {
  static readonly nodeType = "lib.date.Now";
  static readonly title = "Now";
  static readonly description = "Get the current date and time in UTC timezone.";

  async process(): Promise<Record<string, unknown>> {
    return { output: toDateTimeValue(new Date()) };
  }
}

export class ParseDateLibNode extends BaseNode {
  static readonly nodeType = "lib.date.ParseDate";
  static readonly title = "Parse Date";
  static readonly description = "Parse a date string into a structured Date object.";

  defaults() {
    return { date_string: "", input_format: "%Y-%m-%d" as DateFormat };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = String(inputs.date_string ?? this._props.date_string ?? "");
    const format = String(inputs.input_format ?? this._props.input_format ?? "%Y-%m-%d") as DateFormat;
    return { output: toDateValue(parseDateByFormat(value, format)) };
  }
}

export class ParseDateTimeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.ParseDateTime";
  static readonly title = "Parse Date Time";
  static readonly description = "Parse a date/time string into a structured Datetime object.";

  defaults() {
    return { datetime_string: "", input_format: "%Y-%m-%d" as DateFormat };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = String(inputs.datetime_string ?? this._props.datetime_string ?? "");
    const format = String(inputs.input_format ?? this._props.input_format ?? "%Y-%m-%d") as DateFormat;
    return { output: toDateTimeValue(parseDateByFormat(value, format)) };
  }
}

export class AddTimeDeltaLibNode extends BaseNode {
  static readonly nodeType = "lib.date.AddTimeDelta";
  static readonly title = "Add Time Delta";
  static readonly description = "Add or subtract time from a datetime using specified intervals.";

  defaults() {
    return { input_datetime: toDateTimeValue(new Date()), days: 0, hours: 0, minutes: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const base = toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()));
    const days = Number(inputs.days ?? this._props.days ?? 0);
    const hours = Number(inputs.hours ?? this._props.hours ?? 0);
    const minutes = Number(inputs.minutes ?? this._props.minutes ?? 0);
    const out = new Date(base.getTime() + ((days * 24 + hours) * 60 + minutes) * 60 * 1000);
    return { output: toDateTimeValue(out) };
  }
}

export class DateDifferenceLibNode extends BaseNode {
  static readonly nodeType = "lib.date.DateDifference";
  static readonly title = "Date Difference";
  static readonly description = "Calculate the time difference between two datetimes.";

  defaults() {
    return { start_date: toDateTimeValue(new Date()), end_date: toDateTimeValue(new Date()) };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const start = toDate(inputs.start_date ?? this._props.start_date ?? toDateTimeValue(new Date()));
    const end = toDate(inputs.end_date ?? this._props.end_date ?? toDateTimeValue(new Date()));
    const totalSeconds = Math.trunc((end.getTime() - start.getTime()) / 1000);
    const days = Math.floor(totalSeconds / 86400);
    const rem = totalSeconds - days * 86400;
    const hours = Math.floor(rem / 3600);
    const minutes = Math.floor((rem % 3600) / 60);
    const seconds = rem % 60;
    return { total_seconds: totalSeconds, days, hours, minutes, seconds };
  }
}

export class FormatDateTimeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.FormatDateTime";
  static readonly title = "Format Date Time";
  static readonly description = "Convert a datetime object to a custom formatted string.";

  defaults() {
    return { input_datetime: toDateTimeValue(new Date()), output_format: "%B %d, %Y" as DateFormat };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()));
    const format = String(inputs.output_format ?? this._props.output_format ?? "%B %d, %Y") as DateFormat;
    return { output: formatDate(value, format) };
  }
}

export class GetWeekdayLibNode extends BaseNode {
  static readonly nodeType = "lib.date.GetWeekday";
  static readonly title = "Get Weekday";
  static readonly description = "Get the weekday name or number from a datetime.";

  defaults() {
    return { input_datetime: toDateTimeValue(new Date()), as_name: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()));
    const asName = Boolean(inputs.as_name ?? this._props.as_name ?? true);
    return { output: asName ? value.toLocaleDateString("en-US", { weekday: "long" }) : (value.getDay() + 6) % 7 };
  }
}

export class DateRangeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.DateRange";
  static readonly title = "Date Range";
  static readonly description = "Generate a list of dates between start and end dates with custom intervals.";

  defaults() {
    return { start_date: toDateTimeValue(new Date()), end_date: toDateTimeValue(new Date()), step_days: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const start = toDate(inputs.start_date ?? this._props.start_date ?? toDateTimeValue(new Date()));
    const end = toDate(inputs.end_date ?? this._props.end_date ?? toDateTimeValue(new Date()));
    const stepDays = Number(inputs.step_days ?? this._props.step_days ?? 1);
    const output: DateTimeValue[] = [];

    for (let current = new Date(start); current <= end; current = new Date(current.getTime() + stepDays * 86400000)) {
      output.push(toDateTimeValue(current));
      if (stepDays <= 0) {
        break;
      }
    }
    return { output };
  }
}

export class IsDateInRangeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.IsDateInRange";
  static readonly title = "Is Date In Range";
  static readonly description = "Check if a date falls within a specified range with optional inclusivity.";

  defaults() {
    return {
      check_date: toDateTimeValue(new Date()),
      start_date: toDateTimeValue(new Date()),
      end_date: toDateTimeValue(new Date()),
      inclusive: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const check = toDate(inputs.check_date ?? this._props.check_date ?? toDateTimeValue(new Date())).getTime();
    const start = toDate(inputs.start_date ?? this._props.start_date ?? toDateTimeValue(new Date())).getTime();
    const end = toDate(inputs.end_date ?? this._props.end_date ?? toDateTimeValue(new Date())).getTime();
    const inclusive = Boolean(inputs.inclusive ?? this._props.inclusive ?? true);
    return { output: inclusive ? start <= check && check <= end : start < check && check < end };
  }
}

export class GetQuarterLibNode extends BaseNode {
  static readonly nodeType = "lib.date.GetQuarter";
  static readonly title = "Get Quarter";
  static readonly description = "Get the quarter number and start/end dates for a given datetime.";

  defaults() {
    return { input_datetime: toDateTimeValue(new Date()) };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()));
    const quarter = Math.floor(value.getMonth() / 3) + 1;
    const quarterStart = new Date(value.getFullYear(), (quarter - 1) * 3, 1, 0, 0, 0, 0);
    const quarterEnd =
      quarter === 4
        ? new Date(value.getFullYear(), 11, 31, 23, 59, 59, 999)
        : new Date(value.getFullYear(), quarter * 3, 0, 23, 59, 59, 999);

    return {
      quarter,
      quarter_start: toDateTimeValue(quarterStart),
      quarter_end: toDateTimeValue(quarterEnd),
    };
  }
}

export class DateToDatetimeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.DateToDatetime";
  static readonly title = "Date To Datetime";
  static readonly description = "Convert a Date object to a Datetime object at midnight.";

  defaults() {
    return { input_date: toDateValue(new Date()) };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const base = toDate(inputs.input_date ?? this._props.input_date ?? toDateValue(new Date()));
    base.setHours(0, 0, 0, 0);
    return { output: toDateTimeValue(base) };
  }
}

export class DatetimeToDateLibNode extends BaseNode {
  static readonly nodeType = "lib.date.DatetimeToDate";
  static readonly title = "Datetime To Date";
  static readonly description = "Convert a Datetime object to a Date object, removing time component.";

  defaults() {
    return { input_datetime: toDateTimeValue(new Date()) };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: toDateValue(toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()))),
    };
  }
}

export class RelativeTimeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.RelativeTime";
  static readonly title = "Relative Time";
  static readonly description = "Get datetime relative to current time (past or future) with configurable units.";

  defaults() {
    return { amount: 1, unit: "days" as TimeUnit, direction: "future" as TimeDirection };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const amount = Number(inputs.amount ?? this._props.amount ?? 1);
    const unit = String(inputs.unit ?? this._props.unit ?? "days") as TimeUnit;
    const direction = String(inputs.direction ?? this._props.direction ?? "future") as TimeDirection;
    const sign = direction === "past" ? -1 : 1;
    const current = new Date();

    if (unit === "hours") {
      return { output: toDateTimeValue(new Date(current.getTime() + sign * amount * 3600000)) };
    }
    if (unit === "days") {
      return { output: toDateTimeValue(new Date(current.getTime() + sign * amount * 86400000)) };
    }

    let year = current.getUTCFullYear();
    let month = current.getUTCMonth() + 1 + sign * amount;
    while (month <= 0) {
      month += 12;
      year -= 1;
    }
    while (month > 12) {
      month -= 12;
      year += 1;
    }

    const day = current.getUTCDate();
    const maxDay = new Date(Date.UTC(year, month, 0)).getUTCDate();
    if (day > maxDay) {
      throw new Error("day is out of range for month");
    }

    const out = new Date(
      Date.UTC(
        year,
        month - 1,
        day,
        current.getUTCHours(),
        current.getUTCMinutes(),
        current.getUTCSeconds(),
        current.getUTCMilliseconds()
      )
    );
    return { output: toDateTimeValue(out) };
  }
}

export class BoundaryTimeLibNode extends BaseNode {
  static readonly nodeType = "lib.date.BoundaryTime";
  static readonly title = "Boundary Time";
  static readonly description = "Get the start or end boundary of a time period (day, week, month, year).";

  defaults() {
    return {
      input_datetime: toDateTimeValue(new Date()),
      period: "day" as PeriodType,
      boundary: "start" as BoundaryType,
      start_monday: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dt = toDate(inputs.input_datetime ?? this._props.input_datetime ?? toDateTimeValue(new Date()));
    const period = String(inputs.period ?? this._props.period ?? "day") as PeriodType;
    const boundary = String(inputs.boundary ?? this._props.boundary ?? "start") as BoundaryType;
    const startMonday = Boolean(inputs.start_monday ?? this._props.start_monday ?? true);

    const out = new Date(dt);
    if (period === "day") {
      if (boundary === "start") out.setHours(0, 0, 0, 0);
      else out.setHours(23, 59, 59, 999);
      return { output: toDateTimeValue(out) };
    }

    if (period === "week") {
      const weekday = startMonday ? (out.getDay() + 6) % 7 : out.getDay();
      if (boundary === "start") {
        out.setDate(out.getDate() - weekday);
        out.setHours(0, 0, 0, 0);
      } else {
        out.setDate(out.getDate() + (6 - weekday));
        out.setHours(23, 59, 59, 999);
      }
      return { output: toDateTimeValue(out) };
    }

    if (period === "month") {
      if (boundary === "start") {
        out.setDate(1);
        out.setHours(0, 0, 0, 0);
      } else {
        out.setMonth(out.getMonth() + 1, 0);
        out.setHours(23, 59, 59, 999);
      }
      return { output: toDateTimeValue(out) };
    }

    if (boundary === "start") {
      out.setMonth(0, 1);
      out.setHours(0, 0, 0, 0);
    } else {
      out.setMonth(11, 31);
      out.setHours(23, 59, 59, 999);
    }
    return { output: toDateTimeValue(out) };
  }
}

export const LIB_DATE_NODES = [
  TodayLibNode,
  NowLibNode,
  ParseDateLibNode,
  ParseDateTimeLibNode,
  AddTimeDeltaLibNode,
  DateDifferenceLibNode,
  FormatDateTimeLibNode,
  GetWeekdayLibNode,
  DateRangeLibNode,
  IsDateInRangeLibNode,
  GetQuarterLibNode,
  DateToDatetimeLibNode,
  DatetimeToDateLibNode,
  RelativeTimeLibNode,
  BoundaryTimeLibNode,
] as const;
