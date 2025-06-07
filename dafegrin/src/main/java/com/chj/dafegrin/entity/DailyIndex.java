package com.chj.dafegrin.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;

@Entity
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "daily_index")
public class DailyIndex {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "daily_index_id")
    private Long id;

    @Column(name = "daily_index_date")
    private LocalDate date;

    private Double kospi;

    private boolean isPos;

    private Double sentiment;
}
