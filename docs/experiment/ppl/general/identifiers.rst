===========
Identifiers
===========

.. rubric:: Table of contents

.. contents::
   :local:
   :depth: 2


Introduction
============

Identifiers are used for naming your database objects, such as index name, field name, alias etc. Basically there are two types of identifiers: regular identifiers and delimited identifiers.


Regular Identifiers
===================

Description
-----------

A regular identifier is a string of characters that must start with ASCII letter (lower or upper case). The subsequent character can be a combination of letter, digit, underscore (``_``). It cannot be a reversed key word. And whitespace and other special characters are not allowed.

For OpenSearch, the following identifiers are supported extensionally:

1. Identifiers prefixed by dot ``.``: this is called hidden index in OpenSearch, for example ``.opensearch_dashboards``.
2. Identifiers prefixed by at sign ``@``: this is common for meta fields generated in Logstash ingestion.
3. Identifiers with ``-`` in the middle: this is mostly the case for index name with date information.
4. Identifiers with star ``*`` present: this is mostly an index pattern for wildcard match.

Index name with date suffix separated by dash or dots, such as ``cwl-2020.01.11`` or ``logs-7.0-2020.01.11``, is common for those created by Logstash or FileBeat ingestion. So, this kind of identifier used as index name is also supported without the need of being quoted for user convenience. In this case, wildcard within date pattern is also allowed to search for data across indices of different date range. For example, you can use ``logs-2020.1*`` to search in indices for October, November and December 2020.

Examples
--------

Here are examples for using index pattern directly without quotes::

    os> source=accounts | fields account_number, firstname, lastname;
    fetched rows / total rows = 4/4
    +------------------+-------------+------------+
    | account_number   | firstname   | lastname   |
    |------------------+-------------+------------|
    | 1                | Amber       | Duke       |
    | 6                | Hattie      | Bond       |
    | 13               | Nanette     | Bates      |
    | 18               | Dale        | Adams      |
    +------------------+-------------+------------+


Delimited Identifiers
=====================

Description
-----------

A delimited identifier is an identifier enclosed in back ticks `````. In this case, the identifier enclosed is not necessarily a regular identifier. In other words, it can contain any special character not allowed by regular identifier.

Use Cases
---------

Here are typical examples of the use of delimited identifiers:

1. Identifiers of reserved key word name
2. Identifiers with dot ``.`` present: similarly as ``-`` in index name to include date information, it is required to be quoted so parser can differentiate it from identifier with qualifiers.
3. Identifiers with other special character: OpenSearch has its own rule which allows more special character, for example Unicode character is supported in index name.

Examples
--------

Here are examples for quoting an index name by back ticks::

    os> source=`accounts` | fields `account_number`;
    fetched rows / total rows = 4/4
    +------------------+
    | account_number   |
    |------------------|
    | 1                |
    | 6                |
    | 13               |
    | 18               |
    +------------------+


Case Sensitivity
================

Description
-----------

Identifiers are treated in case sensitive manner. So it must be exactly same as what is stored in OpenSearch.

Examples
--------

For example, if you run ``source=Accounts``, it will end up with an index not found exception from our plugin because the actual index name is under lower case.