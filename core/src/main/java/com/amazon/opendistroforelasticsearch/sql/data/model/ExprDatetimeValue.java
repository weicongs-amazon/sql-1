/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package com.amazon.opendistroforelasticsearch.sql.data.model;

import com.amazon.opendistroforelasticsearch.sql.data.type.ExprCoreType;
import com.amazon.opendistroforelasticsearch.sql.data.type.ExprType;
import com.amazon.opendistroforelasticsearch.sql.exception.SemanticCheckException;
import com.google.common.base.Objects;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ExprDatetimeValue extends AbstractExprValue {
  private final LocalDateTime datetime;

  private static final DateTimeFormatter FORMATTER_VARIABLE_MICROS;
  private static final int MIN_FRACTION_SECONDS = 0;
  private static final int MAX_FRACTION_SECONDS = 6;

  static {
    FORMATTER_VARIABLE_MICROS = new DateTimeFormatterBuilder()
        .appendPattern("yyyy-MM-dd HH:mm:ss")
        .appendFraction(
            ChronoField.MICRO_OF_SECOND,
            MIN_FRACTION_SECONDS,
            MAX_FRACTION_SECONDS,
            true)
        .toFormatter();
  }

  /**
   * Constructor with datetime string as input.
   */
  public ExprDatetimeValue(String datetime) {
    try {
      this.datetime = LocalDateTime.parse(datetime, FORMATTER_VARIABLE_MICROS);
    } catch (DateTimeParseException e) {
      throw new SemanticCheckException(String.format("datetime:%s in unsupported format, please "
          + "use yyyy-MM-dd HH:mm:ss[.SSSSSS]", datetime));
    }
  }

  @Override
  public LocalDateTime datetimeValue() {
    return datetime;
  }

  @Override
  public LocalDate dateValue() {
    return datetime.toLocalDate();
  }

  @Override
  public LocalTime timeValue() {
    return datetime.toLocalTime();
  }

  @Override
  public Instant timestampValue() {
    return ZonedDateTime.of(datetime, ZoneId.of("UTC")).toInstant();
  }

  @Override
  public int compare(ExprValue other) {
    return datetime.compareTo(other.datetimeValue());
  }

  @Override
  public boolean equal(ExprValue other) {
    return datetime.equals(other.datetimeValue());
  }

  @Override
  public String value() {
    return String.format("%s %s", DateTimeFormatter.ISO_DATE.format(datetime),
        DateTimeFormatter.ISO_TIME.format((datetime.getNano() == 0)
            ? datetime.truncatedTo(ChronoUnit.SECONDS) : datetime));
  }

  @Override
  public ExprType type() {
    return ExprCoreType.DATETIME;
  }

  @Override
  public String toString() {
    return String.format("DATETIME '%s'", value());
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(datetime);
  }
}
